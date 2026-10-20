"""從刊登資訊估 PSA 分數機率（樂觀 / 基準 / 悲觀三組）

步驟：
1. 依 Mercari 商品狀態選起點：未使用〜目立った傷なし 用「基準」為中心，
   やや傷〜傷や汚れあり 用「悲觀」為中心，全体的に状態が悪い 直接跳過。
2. 讀賣家描述的瑕疵字眼（會排除「傷なし」這類否定句）。
3. 套用照片觀察（置中、邊角白邊、表面、折痕），依 PSA 標準給上限或往下移。
4. 照片不足（少於 2 張、沒有背面）時往悲觀方向混合，代表不確定性。

賣家的正面字眼（美品、完美品）一律不加分：日版描述常偏寬鬆，以照片為準。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ev_calc import GRADES, normalize
from image_analysis import Observation

CONDITION_LABELS = {
    1: "新品、未使用",
    2: "未使用に近い",
    3: "目立った傷や汚れなし",
    4: "やや傷や汚れあり",
    5: "傷や汚れあり",
    6: "全体的に状態が悪い",
}

# 商品狀態 -> 中心情境；None = 跳過
CONDITION_CENTER = {1: "base", 2: "base", 3: "base", 4: "pessimistic", 5: "pessimistic", 6: None}

# PSA 正面置中標準（較寬側 %）-> 可得最高分
CENTERING_CAPS = [(55, 10), (60, 9), (65, 8), (70, 7), (80, 6), (90, 5)]

# 描述字眼：(regex, 動作, 說明)。動作 cap=N 上限 PSA N；shift=N 整體往下 N 級
DESC_RULES = [
    (r"折れ|折り目|折れ目|しわ|シワ", ("cap", 5), "描述提到折痕／皺"),
    (r"凹み|へこみ|ヘコみ|打痕", ("cap", 6), "描述提到凹痕"),
    (r"角潰れ|角つぶれ|角折れ|(?<!白)欠け", ("cap", 7), "描述提到角損／缺角"),
    (r"白欠け|白かけ|白カケ|白いかけ|エッジ.?白|角.?白", ("cap", 8), "描述提到白邊"),
    (r"初期傷|スレ|擦れ|小傷|微傷|キズ|傷", ("shift", 1), "描述提到刮痕／擦傷"),
    (r"センターズレ|センタリング|印刷ズレ|ズレ", ("cap", 8), "描述提到置中或印刷偏移"),
    (r"汚れ|シミ|しみ|ヤケ|日焼け|変色", ("shift", 1), "描述提到污漬／變色"),
]
NEGATION = re.compile(r"^[^。\n、,.!！]{0,6}?(なし|無し|ない|無い|ありません|ございません|見当たりません|ナシ)")


def condition_id(text: str | int | None) -> int | None:
    if isinstance(text, int):
        return text
    if not text:
        return None
    for cid, label in CONDITION_LABELS.items():
        if label in text or label.replace("、", "") in text:
            return cid
    if "新品" in text:
        return 1
    return None


def shift_down(probs: dict, steps: int) -> dict:
    """每一級的機率整體往下移 steps 級（PSA 10 的機率移到 9，依此類推）"""
    probs = normalize(probs)
    out = {g: 0.0 for g in GRADES}
    for i, g in enumerate(GRADES):
        out[GRADES[min(i + steps, len(GRADES) - 1)]] += probs[g]
    return out


def cap_at(probs: dict, max_grade) -> dict:
    """高於上限的機率全部壓到上限那一級"""
    probs = normalize(probs)
    idx = GRADES.index(max_grade)
    out = {g: 0.0 for g in GRADES}
    for i, g in enumerate(GRADES):
        out[GRADES[max(i, idx)]] += probs[g]
    return out


def blend(a: dict, b: dict, w: float) -> dict:
    a, b = normalize(a), normalize(b)
    return {g: a[g] * (1 - w) + b[g] * w for g in GRADES}


@dataclass
class Estimate:
    scenarios: dict | None  # {"optimistic","base","pessimistic"} -> probs；None = 跳過
    condition: int | None
    notes: list = field(default_factory=list)
    observations: list = field(default_factory=list)


def _desc_findings(text: str) -> list:
    found = []
    for pattern, action, label in DESC_RULES:
        for m in re.finditer(pattern, text or ""):
            if NEGATION.match(text[m.end():]):
                continue  # 「傷なし」「折れはありません」
            found.append((action, f"{label}（「{m.group(0)}」）"))
            break
    return found


def _merge_observations(observations: list, min_conf: float) -> Observation:
    """多個 analyzer 的結果合併：信心高的優先，低於門檻的欄位不採用"""
    merged = Observation(source="+".join(o.source for o in observations))
    for obs in sorted(observations, key=lambda o: o.confidence):
        if obs.confidence < min_conf:
            continue
        for f in ("centering_lr", "centering_tb", "corners", "edges", "surface", "crease", "has_front", "has_back", "authenticity_concern"):
            v = getattr(obs, f)
            if v is not None:
                setattr(merged, f, v)
        merged.confidence = max(merged.confidence, obs.confidence)
    # 照片張數相關的旗標即使信心低也可以參考
    for obs in observations:
        if merged.has_back is None and obs.has_back is not None:
            merged.has_back = obs.has_back
    return merged


def estimate(
    base_sets: dict,
    condition: str | int | None,
    description: str = "",
    title: str = "",
    observations: list | None = None,
    photo_count: int | None = None,
    min_confidence: float = 0.5,
) -> Estimate:
    """base_sets: 卡種設定的三組機率 {"optimistic","base","pessimistic"}"""
    cid = condition_id(condition)
    notes = []
    center = CONDITION_CENTER.get(cid, "pessimistic") if cid else "pessimistic"
    if cid is None:
        notes.append("看不到商品狀態，用悲觀情境")
    if center is None:
        return Estimate(None, cid, [f"商品狀態「{CONDITION_LABELS[cid]}」直接跳過"])
    notes.append(f"商品狀態「{CONDITION_LABELS.get(cid, '?')}」→ 以{'基準' if center == 'base' else '悲觀'}為中心")

    opt, base, pess = (normalize(base_sets[k]) for k in ("optimistic", "base", "pessimistic"))
    if center == "base":
        scen = {"optimistic": opt, "base": base, "pessimistic": pess}
    else:
        scen = {"optimistic": base, "base": pess, "pessimistic": shift_down(pess, 1)}

    def apply(fn, *args):
        for k in scen:
            scen[k] = fn(scen[k], *args)

    # 描述瑕疵
    for (kind, n), label in _desc_findings(f"{title}\n{description}"):
        notes.append(f"{label} → {'上限 PSA ' + str(n) if kind == 'cap' else '往下 ' + str(n) + ' 級'}")
        apply(cap_at if kind == "cap" else shift_down, n)

    # 照片觀察
    observations = observations or []
    for o in observations:
        notes.extend(f"[{o.source}] {n}" for n in o.notes)
    obs = _merge_observations(observations, min_confidence) if observations else None
    if obs:
        worst_center = max([c for c in (obs.centering_lr, obs.centering_tb) if c is not None], default=None)
        if worst_center is not None:
            cap = next((g for limit, g in CENTERING_CAPS if worst_center <= limit), "low")
            if cap != 10:
                notes.append(f"置中約 {worst_center:.0f}/{100 - worst_center:.0f} → 上限 PSA {cap}")
                apply(cap_at, cap)
        sev = [s for s in (obs.corners, obs.edges, obs.surface) if s]
        if "heavy" in sev:
            notes.append("照片看得到明顯邊角／表面瑕疵 → 上限 PSA 7")
            apply(cap_at, 7)
        elif "light" in sev:
            notes.append("照片看得到輕微白點／刮痕 → 上限 PSA 9 並往下混合")
            apply(cap_at, 9)
            for k in scen:
                scen[k] = blend(scen[k], shift_down(scen[k], 1), 0.5)
        if obs.crease:
            notes.append("照片看得到折痕 → 上限 PSA 5")
            apply(cap_at, 5)
        if obs.authenticity_concern:
            notes.append("⚠️ 照片有仿品疑慮，請人工確認")

    # 照片不足 -> 不確定性往悲觀混
    missing_back = obs is not None and obs.has_back is False
    if (photo_count is not None and photo_count < 2) or missing_back:
        notes.append("照片不足（張數少或沒有背面）→ 往悲觀混合 50%")
        for k in scen:
            scen[k] = blend(scen[k], shift_down(scen[k], 1), 0.5)

    return Estimate(scen, cid, notes, observations)
