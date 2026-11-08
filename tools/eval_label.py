"""拿人工標註當考卷，比較不同的 Claude 看圖標瑕疵做法，挑出最準的

用法：python tools/eval_label.py EVAL_DIR --variant full,grid,tiles [--effort medium] [--limit N]
  EVAL_DIR/img/<id>.jpg     照片
  EVAL_DIR/gt/<id>.json     標註頁的人工結果 {marks:[{t, box}], listing:{title, condition}}
  → EVAL_DIR/pred/<variant>/<id>.json   模型結果（已存在就不重打 API，方便只重算分數）
  → 印出每個做法的漏標、誤標、召回率、精確率，並寫 EVAL_DIR/pred/<variant>/score.json

比對：AI 框和人工框各外擴 PAD（照片短邊比例）後有重疊就算對到同一個瑕疵（不管類型）。
- 召回率 = 被 AI 對到的人工瑕疵 / 人工瑕疵
- 精確率 = 對到人工瑕疵的 AI 框 / AI 框
- 乾淨判斷：人工沒標任何瑕疵的照片，AI 也沒標 → 算對
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import anthropic
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from auto_label import MARK_TYPES, PROMPT, SCHEMA, label_photo  # noqa: E402

PAD = 0.02
MODEL = "claude-opus-5-5"

TYPE_HELP = """類型：
- edge：邊緣白邊（卡邊的白色紙芯露出）
- corner：角尖白掉或磨圓
- spot：白點、小缺損
- scratch：刮痕（閃卡面上的細線）
- crease：摺痕、壓摺線
- dent：壓痕、凹陷
- stain：髒污、變色"""

GRID_PROMPT = """這是日本 Mercari 上一張寶可夢卡片的賣家照片，要送 PSA 鑑定前先看卡況。
刊登標題：{title}
賣家標示品況：{condition}

照片外圍加了白色尺規：上下標 x、左右標 y，刻度 0~100，照片上也畫了每 10 一格的淡格線。
請只看卡片本身的實體瑕疵，忽略背景、桌面、反光、卡套、壓縮雜訊，也忽略印刷本來就有的白線、白框、文字。
每一處瑕疵給一個框，用尺規刻度讀出 x0,y0,x1,y1（0~100，照片左上是 0,0），先找到瑕疵附近的格線再讀刻度。
{types}
框要緊貼瑕疵（通常只有幾格寬）；同一條邊上連續的白邊給一個框。
寧可多看幾遍再下結論：把卡片四個角、四條邊、表面依序檢查一次。
side：正面 front、背面 back、看不出 other。clean：看得到的部分完全沒有瑕疵。psa：看得到整張卡才估 1~10，否則 0。"""

TILE_PROMPT = """這是日本 Mercari 上一張寶可夢卡片的賣家照片，要送 PSA 鑑定前先看卡況。
刊登標題：{title}
賣家標示品況：{condition}

第一張圖是整張照片（給你看全貌），後面 {n} 張是卡片的放大區塊，編號 tile 0~{last}，每張放大圖左上角寫了編號。
請在放大區塊上找卡片本身的實體瑕疵，忽略背景、桌面、反光、卡套、壓縮雜訊，也忽略印刷本來就有的白線、白框、文字。
每一處瑕疵回報它在哪個 tile，以及在那張放大圖裡的框 x0,y0,x1,y1（0~1，相對那張放大圖）。
同一個瑕疵出現在兩個重疊的區塊時，只回報一次。
{types}
框要緊貼瑕疵；同一條邊上連續的白邊給一個框。
side：正面 front、背面 back、看不出 other。clean：看得到的部分完全沒有瑕疵。psa：看得到整張卡才估 1~10，否則 0。"""


def _schema_with(extra: dict | None = None) -> dict:
    s = json.loads(json.dumps(SCHEMA))
    if extra:
        s["properties"]["marks"]["items"]["properties"].update(extra)
        s["properties"]["marks"]["items"]["required"] = list(s["properties"]["marks"]["items"]["required"]) + list(extra)
    return s


def _jpeg(im: Image.Image, q: int = 90) -> dict:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=q)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(buf.getvalue()).decode()}}


def _call(client, content, schema, effort) -> tuple[dict, dict]:
    resp = client.beta.messages.create(
        model=MODEL, max_tokens=6000,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": content}],
    )
    if resp.stop_reason == "refusal":
        return {"marks": [], "side": "other", "clean": False, "psa": 0, "note": "refused"}, {}
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    return json.loads(text), {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}


def _font(size: int):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def grid_image(im: Image.Image) -> tuple[Image.Image, int]:
    w, h = im.size
    m = 44
    out = Image.new("RGB", (w + 2 * m, h + 2 * m), "white")
    out.paste(im, (m, m))
    d = ImageDraw.Draw(out, "RGBA")
    f = _font(18)
    for i in range(0, 101, 10):
        x = m + w * i / 100
        y = m + h * i / 100
        d.line([(x, m), (x, m + h)], fill=(255, 255, 255, 70), width=1)
        d.line([(m, y), (m + w, y)], fill=(255, 255, 255, 70), width=1)
        for yy in (2, m + h + 18):
            d.text((x, yy), str(i), fill="black", font=f, anchor="mt" if yy == 2 else "mm")
        d.text((m - 6, y), str(i), fill="black", font=f, anchor="rm")
        d.text((m + w + 6, y), str(i), fill="black", font=f, anchor="lm")
    for i in range(0, 101, 5):  # 小刻度
        x = m + w * i / 100
        y = m + h * i / 100
        d.line([(x, m - 6), (x, m)], fill="black", width=1)
        d.line([(x, m + h), (x, m + h + 6)], fill="black", width=1)
        d.line([(m - 6, y), (m, y)], fill="black", width=1)
        d.line([(m + w, y), (m + w + 6, y)], fill="black", width=1)
    return out, m


def tiles_of(w: int, h: int, n: int = 3, overlap: float = 0.2) -> list[tuple[float, float, float, float]]:
    """n×n 重疊區塊（正規化座標）"""
    step = 1 / (n - (n - 1) * overlap)
    size = step
    out = []
    for r in range(n):
        for c in range(n):
            x0 = c * size * (1 - overlap)
            y0 = r * size * (1 - overlap)
            out.append((x0, y0, min(1, x0 + size), min(1, y0 + size)))
    return out


def card_region(raw: bytes, margin: float = 0.03) -> tuple[float, float, float, float]:
    """OpenCV 找卡片外框（正規化座標，外擴 margin）；找不到就回整張"""
    import cv2
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import defect_scan
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    h, w = bgr.shape[:2]
    loc = defect_scan.locate_card(bgr)
    if loc is None:
        return (0.0, 0.0, 1.0, 1.0)
    x, y, bw, bh = cv2.boundingRect(loc[1])
    if bw * bh < 0.08 * w * h:
        return (0.0, 0.0, 1.0, 1.0)
    return (max(0.0, x / w - margin), max(0.0, y / h - margin), min(1.0, (x + bw) / w + margin), min(1.0, (y + bh) / h + margin))


def _clamp_box(x0, y0, x1, y1):
    x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in (x0, y0, x1, y1))
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def run_variant(client, variant: str, raw: bytes, listing: dict, effort: str) -> dict:
    title, cond = listing.get("title", ""), listing.get("condition", "")
    if variant == "full":
        d = label_photo(client, raw, listing, MODEL, effort)
        return d
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = im.size
    if variant == "grid":
        g, _ = grid_image(im)
        d, usage = _call(client, [_jpeg(g), {"type": "text", "text": GRID_PROMPT.format(title=title, condition=cond, types=TYPE_HELP)}],
                         _schema_with(), effort)
        for m in d["marks"]:
            m["box"] = _clamp_box(*(m.pop(k) / 100 for k in ("x0", "y0", "x1", "y1")))
        d["usage"] = usage
        return d
    if variant.startswith("tiles") or variant.startswith("card"):
        # tilesN：整張照片切 N×N；cardN：先用 OpenCV 找卡片範圍，只在卡片上切 N×N（card1 = 只放大卡片）
        r = card_region(raw) if variant.startswith("card") else (0.0, 0.0, 1.0, 1.0)
        n = int(variant[5:] or 3) if variant.startswith("tiles") else int(variant[4:] or 2)
        ts = [(r[0] + a * (r[2] - r[0]), r[1] + b * (r[3] - r[1]), r[0] + c * (r[2] - r[0]), r[1] + e * (r[3] - r[1]))
              for a, b, c, e in tiles_of(w, h, n)]
        content = [_jpeg(im)]
        f = _font(28)
        for i, (x0, y0, x1, y1) in enumerate(ts):
            t = im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
            sc = 1000 / max(t.size)
            t = t.resize((int(t.width * sc), int(t.height * sc)), Image.LANCZOS)
            dd = ImageDraw.Draw(t)
            dd.rectangle([0, 0, 120, 40], fill="white")
            dd.text((6, 4), f"tile {i}", fill="black", font=f)
            content.append(_jpeg(t))
        content.append({"type": "text", "text": TILE_PROMPT.format(title=title, condition=cond, n=len(ts), last=len(ts) - 1, types=TYPE_HELP)})
        d, usage = _call(client, content, _schema_with({"tile": {"type": "integer"}}), effort)
        for m in d["marks"]:
            ti = min(len(ts) - 1, max(0, int(m.pop("tile"))))
            tx0, ty0, tx1, ty1 = ts[ti]
            b = _clamp_box(m.pop("x0"), m.pop("y0"), m.pop("x1"), m.pop("y1"))
            m["box"] = [tx0 + b[0] * (tx1 - tx0), ty0 + b[1] * (ty1 - ty0), tx0 + b[2] * (tx1 - tx0), ty0 + b[3] * (ty1 - ty0)]
            m["tile"] = ti
        d["usage"] = usage
        return d
    raise SystemExit(f"不認識的做法 {variant}")


def _grow(b, pad):
    return [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad]


def _overlap(a, b) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def score(gt_dir: Path, pred_dir: Path) -> dict:
    tot_h = hit_h = tot_a = hit_a = 0
    clean_ok = clean_n = 0
    flagged_ok = flagged_n = 0
    per = []
    for g in sorted(gt_dir.glob("*.json")):
        p = pred_dir / g.name
        if not p.exists():
            continue
        gt = json.loads(g.read_text())
        pr = json.loads(p.read_text())
        w, h = gt.get("w") or 1, gt.get("h") or 1
        px, py = PAD * min(w, h) / w, PAD * min(w, h) / h
        hb = [_grow(m["box"], 0) for m in gt["marks"]]
        hb = [[b[0] - px, b[1] - py, b[2] + px, b[3] + py] for b in hb]
        ab = [m["box"] for m in pr.get("marks", [])]
        ab = [[b[0] - px, b[1] - py, b[2] + px, b[3] + py] for b in ab]
        hh = sum(any(_overlap(x, y) for y in ab) for x in hb)
        ah = sum(any(_overlap(x, y) for y in hb) for x in ab)
        tot_h += len(hb); hit_h += hh; tot_a += len(ab); hit_a += ah
        if not hb:
            clean_n += 1; clean_ok += not ab
        else:
            flagged_n += 1; flagged_ok += bool(ab)
        per.append({"id": g.stem, "human": len(hb), "ai": len(ab), "human_hit": hh, "ai_hit": ah})
    return {
        "photos": len(per), "human_marks": tot_h, "ai_marks": tot_a,
        "recall": round(hit_h / tot_h, 3) if tot_h else None,
        "precision": round(hit_a / tot_a, 3) if tot_a else None,
        "missed": tot_h - hit_h, "false": tot_a - hit_a,
        "clean_photo_acc": f"{clean_ok}/{clean_n}", "flagged_photo_acc": f"{flagged_ok}/{flagged_n}",
        "per_photo": per,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", type=Path)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    client = None if a.score_only else anthropic.Anthropic()
    gts = sorted((a.eval_dir / "gt").glob("*.json"))
    if a.limit:
        gts = gts[: a.limit]
    summary = {}
    for v in a.variant.split(","):
        tag = v if a.effort == "medium" else f"{v}-{a.effort}"
        out = a.eval_dir / "pred" / tag
        out.mkdir(parents=True, exist_ok=True)
        tin = tout = 0
        for g in gts if client else []:
            dst = out / g.name
            if dst.exists():
                continue
            gt = json.loads(g.read_text())
            raw = (a.eval_dir / "img" / f"{g.stem}.jpg").read_bytes()
            try:
                d = run_variant(client, v, raw, gt.get("listing", {}), a.effort)
            except anthropic.APIStatusError as e:
                print(f"{tag} {g.stem}: API 錯誤 {e.status_code} {e.message}", file=sys.stderr)
                continue
            tin += d.get("usage", {}).get("in", 0); tout += d.get("usage", {}).get("out", 0)
            dst.write_text(json.dumps(d, ensure_ascii=False))
            print(f"{tag} {g.stem}: marks={len(d['marks'])} human={len(gt['marks'])}", flush=True)
        s = score(a.eval_dir / "gt", out)
        s["tokens"] = {"in": tin, "out": tout}
        (out / "score.json").write_text(json.dumps(s, ensure_ascii=False, indent=1))
        summary[tag] = {k: v2 for k, v2 in s.items() if k != "per_photo"}
    print("\n=== 成績 ===")
    for tag, s in summary.items():
        print(tag, json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
