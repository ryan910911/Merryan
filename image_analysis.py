"""從刊登照片看卡況（可插拔）

每個 analyzer 吃「照片 bytes 清單 + 刊登資料」，回傳 Observation。
grade_estimator 再依 Observation 調整分數機率。

內建兩種：
- heuristic：純像素量測（numpy + Pillow），量正面黃框置中、背面藍底邊角白邊。
  只在照片夠正、卡片輪廓清楚時才給高信心；手機斜拍、卡套反光多半會被判低信心而不採用。
- claude：把照片交給 Claude 視覺模型判讀（需要 ANTHROPIC_API_KEY，會產生 API 費用）。

要接別的模型，寫一個有 `name` 和 `analyze(images, listing) -> Observation` 的類別，
加進 ANALYZERS 即可。
"""
from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

CARD_ASPECT = 63 / 88  # 寶可夢卡寬高比
SEVERITY = ("none", "light", "heavy")


@dataclass
class Observation:
    source: str
    confidence: float = 0.0  # 0~1，低於門檻的觀察不會拿來調機率
    centering_lr: float | None = None  # 正面左右置中「較寬那側」百分比，例如 60 代表 60/40
    centering_tb: float | None = None  # 正面上下
    corners: str | None = None  # none / light / heavy
    edges: str | None = None
    surface: str | None = None  # 刮痕、壓痕、印刷線
    crease: bool | None = None  # 折痕
    has_front: bool | None = None
    has_back: bool | None = None
    authenticity_concern: bool | None = None
    notes: list = field(default_factory=list)


def _load(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img.thumbnail((1200, 1200))
    return np.asarray(img).astype(np.float32) / 255.0


def _hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn + 1e-6
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    s = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0)
    return h, s, mx


def find_card_box(rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """以四邊背景色為基準，找出與背景差異大的最大區塊外框（假設卡片大致正放）"""
    h, w, _ = rgb.shape
    border = np.concatenate([rgb[:4].reshape(-1, 3), rgb[-4:].reshape(-1, 3), rgb[:, :4].reshape(-1, 3), rgb[:, -4:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    diff = np.abs(rgb - bg).sum(-1)
    mask = diff > 0.25
    rows = np.where(mask.mean(1) > 0.2)[0]
    cols = np.where(mask.mean(0) > 0.2)[0]
    if len(rows) < h * 0.3 or len(cols) < w * 0.2:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _yellow_mask(rgb: np.ndarray) -> np.ndarray:
    h, s, v = _hsv(rgb)
    return (h > 35) & (h < 65) & (s > 0.35) & (v > 0.5)


def _blue_ratio(rgb: np.ndarray) -> float:
    h, s, v = _hsv(rgb)
    return float(((h > 190) & (h < 250) & (s > 0.3)).mean())


def _run_length(line: np.ndarray) -> int:
    """從線的開頭算連續 True 的長度（容忍 1~2 像素雜訊）"""
    n = 0
    miss = 0
    for v in line:
        if v:
            n += 1 + miss
            miss = 0
        else:
            miss += 1
            if miss > 2:
                break
    return n


def measure_centering(card: np.ndarray) -> tuple[float, float, float] | None:
    """量正面黃框：回傳 (左右較寬側%, 上下較寬側%, 信心)"""
    yellow = _yellow_mask(card)
    if yellow.mean() < 0.05:
        return None  # 沒有黃框（可能是背面或非寶可夢卡）
    h, w = yellow.shape

    def sides(mask: np.ndarray, n: int) -> tuple[list, list]:
        a, b = [], []
        length = mask.shape[0]
        for i in np.linspace(length * 0.3, length * 0.7, n).astype(int):
            a.append(_run_length(mask[i]))
            b.append(_run_length(mask[i][::-1]))
        return a, b

    left, right = sides(yellow, 15)
    top, bottom = sides(yellow.T, 15)

    def ratio(a: list, b: list) -> tuple[float, float]:
        ma, mb = np.median(a), np.median(b)
        if ma + mb == 0:
            return 50.0, 1.0
        spread = (np.std(a) + np.std(b)) / (ma + mb + 1e-6)
        return float(max(ma, mb) / (ma + mb) * 100), float(spread)

    lr, s1 = ratio(left, right)
    tb, s2 = ratio(top, bottom)
    border_frac = (np.median(left) + np.median(right)) / w
    # 黃框應約占寬度 6~14%；超出代表沒量到框
    if not 0.03 < border_frac < 0.25:
        return None
    confidence = float(np.clip(1 - (s1 + s2) * 2, 0, 1))
    return lr, tb, confidence


def measure_whitening(card: np.ndarray) -> tuple[str, float] | None:
    """量背面藍底邊緣與四角的白點比例"""
    if _blue_ratio(card) < 0.3:
        return None
    h, w, _ = card.shape
    band = max(2, int(min(h, w) * 0.02))
    _, s, v = _hsv(card)
    white = (s < 0.18) & (v > 0.8)
    edge = np.concatenate([white[:band].ravel(), white[-band:].ravel(), white[:, :band].ravel(), white[:, -band:].ravel()])
    c = band * 3
    corners = np.concatenate([white[:c, :c].ravel(), white[:c, -c:].ravel(), white[-c:, :c].ravel(), white[-c:, -c:].ravel()])
    score = max(float(edge.mean()), float(corners.mean()))
    if score < 0.02:
        sev = "none"
    elif score < 0.08:
        sev = "light"
    else:
        sev = "heavy"
    return sev, score


class HeuristicAnalyzer:
    name = "heuristic"

    def analyze(self, images: list[bytes], listing: dict) -> Observation:
        obs = Observation(source=self.name)
        best_center = None
        whitening = []
        for raw in images:
            try:
                rgb = _load(raw)
            except Exception:
                continue
            box = find_card_box(rgb)
            if not box:
                continue
            x0, y0, x1, y1 = box
            aspect = (x1 - x0) / max(1, (y1 - y0))
            aspect_ok = abs(aspect - CARD_ASPECT) / CARD_ASPECT < 0.06
            card = rgb[y0:y1, x0:x1]
            c = measure_centering(card)
            if c:
                obs.has_front = True
                lr, tb, conf = c
                if not aspect_ok:
                    conf *= 0.3  # 斜拍或沒裁到整張卡
                if best_center is None or conf > best_center[2]:
                    best_center = (lr, tb, conf)
            wtn = measure_whitening(card)
            if wtn:
                obs.has_back = True
                whitening.append((wtn, 1.0 if aspect_ok else 0.3))
        confs = []
        if best_center:
            obs.centering_lr, obs.centering_tb, conf = best_center
            confs.append(conf)
            obs.notes.append(f"量測置中 左右 {obs.centering_lr:.0f}/{100 - obs.centering_lr:.0f}、上下 {obs.centering_tb:.0f}/{100 - obs.centering_tb:.0f}（信心 {conf:.0%}）")
        if whitening:
            (sev, score), conf = max(whitening, key=lambda x: x[0][1])
            obs.corners = obs.edges = sev
            confs.append(conf * 0.6)  # 照片解析度與反光影響大，打折
            obs.notes.append(f"背面邊角白點比例 {score:.1%}（{sev}）")
        obs.confidence = float(min(confs)) if confs else 0.0
        return obs


VISION_PROMPT = """你是寶可夢卡 PSA 鑑定前的驗卡員。以下是一張 Mercari 二手刊登的照片與賣家描述。
請只根據照片能看到的證據判讀，看不出來的欄位填 null，不要猜。

- centering_lr / centering_tb：正面左右、上下置中，填較寬那一側的百分比（例如 60/40 填 60）。
- corners / edges / surface：none（看不到瑕疵）、light（輕微白點、刮痕）、heavy（明顯缺角、白邊、壓痕）。
- crease：是否看得到折痕。
- has_front / has_back：照片是否包含正面、背面。
- authenticity_concern：印刷、字體、顏色、光澤是否有仿品疑慮。
- confidence：照片清晰度與角度足以支撐你判讀的程度，0 到 1。
- notes：用繁體中文寫 1 到 3 句重點。

刊登標題：{title}
商品狀態：{condition}
賣家描述：
{description}
"""

_SEV = {"anyOf": [{"type": "string", "enum": list(SEVERITY)}, {"type": "null"}]}
_NUM = {"anyOf": [{"type": "number"}, {"type": "null"}]}
_BOOL = {"anyOf": [{"type": "boolean"}, {"type": "null"}]}
VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "centering_lr": _NUM,
        "centering_tb": _NUM,
        "corners": _SEV,
        "edges": _SEV,
        "surface": _SEV,
        "crease": _BOOL,
        "has_front": _BOOL,
        "has_back": _BOOL,
        "authenticity_concern": _BOOL,
        "confidence": {"type": "number"},
        "notes": {"type": "string"},
    },
    "required": list(["centering_lr", "centering_tb", "corners", "edges", "surface", "crease", "has_front", "has_back", "authenticity_concern", "confidence", "notes"]),
    "additionalProperties": False,
}


def _media_type(raw: bytes) -> str:
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


class ClaudeVisionAnalyzer:
    """用 Claude 看照片。沒有 ANTHROPIC_API_KEY 時自動略過。"""

    name = "claude"

    def __init__(self, model: str = "claude-opus-5-5", max_images: int = 6, effort: str = "medium"):
        self.model = model
        self.max_images = max_images
        self.effort = effort

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def analyze(self, images: list[bytes], listing: dict) -> Observation:
        import anthropic

        content = []
        for raw in images[: self.max_images]:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": _media_type(raw), "data": base64.standard_b64encode(raw).decode()},
            })
        content.append({"type": "text", "text": VISION_PROMPT.format(
            title=listing.get("title", ""),
            condition=listing.get("condition", ""),
            description=(listing.get("description") or "")[:3000],
        )})
        client = anthropic.Anthropic()
        resp = client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": VISION_SCHEMA}},
            messages=[{"role": "user", "content": content}],
        )
        if resp.stop_reason == "refusal":
            return Observation(source=self.name, notes=["視覺模型拒絕判讀"])
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        data = json.loads(text)
        notes = data.pop("notes", "")
        obs = Observation(source=self.name, **{k: v for k, v in data.items() if k in Observation.__dataclass_fields__})
        if notes:
            obs.notes.append(notes)
        return obs


def build_analyzers(cfg: dict) -> list:
    """依 cards.yaml 的 analysis 設定建立 analyzer 清單"""
    cfg = cfg or {}
    out = []
    for name in cfg.get("analyzers", ["heuristic", "claude"]):
        if name == "heuristic":
            out.append(HeuristicAnalyzer())
        elif name == "claude":
            if ClaudeVisionAnalyzer.available():
                out.append(ClaudeVisionAnalyzer(
                    model=cfg.get("claude_model", "claude-opus-5-5"),
                    max_images=int(cfg.get("max_images", 6)),
                    effort=cfg.get("claude_effort", "medium"),
                ))
    return out
