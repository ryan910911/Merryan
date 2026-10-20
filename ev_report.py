"""把照片判讀與 EV 結果畫成一張 PNG（推到 Telegram 用）

左邊：所有判讀過的刊登照片（含特寫）排成格狀，標出卡片範圍、白邊／白點（紅圈）、
刮痕與疑似摺痕（線框）、正面黃框內緣與置中比例。
右邊：PSA 各級機率長條圖（樂觀／基準／悲觀三組，同一色系由淺到深）＋ EV 摘要。

圖上文字用英文與數字；有 CJK 字型（GitHub Actions 會裝 fonts-noto-cjk）時才顯示日文標題。
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ev_calc import GRADES, VERDICT_BUY, VERDICT_PASS, VERDICT_RISKY, VERDICT_SKIP

SURFACE = "#fcfcfb"
INK = "#1f1f1e"
INK_2 = "#5f5e5b"
GRID = "#e6e5e1"
# 樂觀 / 基準 / 悲觀：單一藍色系由淺到深（有序資料）
SCEN_COLORS = {"optimistic": "#86b6ef", "base": "#2a78d6", "pessimistic": "#104281"}
SCEN_LABEL = {"optimistic": "Optimistic", "base": "Base", "pessimistic": "Pessimistic"}
GOOD, WARN, SERIOUS, CRIT = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
VERDICT_STYLE = {
    VERDICT_BUY: ("BUY", GOOD),
    VERDICT_RISKY: ("RISKY - pessimistic loses", WARN),
    VERDICT_PASS: ("PASS", CRIT),
    VERDICT_SKIP: ("SKIP", INK_2),
}

CJK_FONTS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]
LATIN_FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"]


def _font_path() -> tuple[str | None, bool]:
    for p in CJK_FONTS:
        if Path(p).exists():
            return p, True
    for p in LATIN_FONTS:
        try:
            ImageFont.truetype(p, 10)
            return p, False
        except OSError:
            continue
    return None, False


FONT_PATH, HAS_CJK = _font_path()


def font(size: int):
    if FONT_PATH:
        return ImageFont.truetype(FONT_PATH, size)
    return ImageFont.load_default(size=size)


def _center_color(worse_side: float) -> str:
    """置中依 PSA 標準：≤55 可拿 10、≤60 最高 9、≤65 最高 8、更差"""
    if worse_side <= 55:
        return GOOD
    if worse_side <= 60:
        return WARN
    if worse_side <= 65:
        return SERIOUS
    return CRIT


SEV_COLOR = {"none": GOOD, "light": WARN, "heavy": CRIT}


def _worst(*levels) -> str:
    order = ["none", "light", "heavy"]
    return max((l for l in levels if l), key=order.index, default="none")


def _scan_level(scan) -> str:
    surface = "heavy" if scan.crease_suspect else "light" if scan.lines or scan.specks >= 3 else "none"
    return _worst(scan.edge_severity, scan.corner_severity, surface)


def has_findings(scan) -> bool:
    return _scan_level(scan) != "none"


def annotate_photo(raw: bytes, annotations: list, height: int | None = None, max_size: tuple | None = None,
                   index: int | None = None) -> Image.Image:
    """把量測結果畫在照片上：綠框 = 偵測到的卡片範圍，紅圈 = 白邊／白點，線框 = 刮痕或疑似摺痕"""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if max_size:
        scale = min(max_size[0] / img.width, max_size[1] / img.height)
    else:
        scale = height / img.height
    img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    d = ImageDraw.Draw(img)
    fsize = 22 if img.width >= 900 else 16 if img.width >= 400 else 13
    f = font(fsize)
    lw = 2 if img.width >= 400 else 1
    tag_y = 4
    for a in annotations:
        if a["kind"] == "front":
            x0, y0, x1, y1 = (v * scale for v in a["box"])
            l, r, t, b = (v * scale for v in a["borders"])
            col = _center_color(max(a["lr"], a["tb"]))
            d.rectangle([x0 + l, y0 + t, x1 - r, y1 - b], outline=INK, width=lw + 3)  # 深色底線，黃框上也看得到
            d.rectangle([x0 + l, y0 + t, x1 - r, y1 - b], outline=col, width=lw)
            label = f"L/R {a['lr']:.0f}/{100 - a['lr']:.0f}  T/B {a['tb']:.0f}/{100 - a['tb']:.0f}"
            if a.get("confidence", 1) < 0.5:
                label += " (low conf.)"
            _tag(d, (4, img.height - fsize - 10), label, col, f)
        elif a["kind"] == "scan":
            sc = a["scan"]
            if sc.hull is not None:
                pts = [(float(x) * scale, float(y) * scale) for x, y in sc.hull.reshape(-1, 2)]
                d.line(pts + pts[:1], fill=GOOD, width=lw)
            if sc.defect_mask is not None and sc.defect_mask.any():
                import cv2
                import numpy as np

                k = max(3, int(min(sc.defect_mask.shape) * 0.01))
                blobs = cv2.dilate(sc.defect_mask, np.ones((k, k), np.uint8))
                contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                m = sc.scale * scale
                for c in contours:
                    (cx, cy), rad = cv2.minEnclosingCircle(c)
                    rad = rad * m + 6
                    d.ellipse([cx * m - rad, cy * m - rad, cx * m + rad, cy * m + rad], outline=CRIT, width=lw + 1)
            for line in sc.lines:
                col = CRIT if line["kind"] == "crease" else SERIOUS
                for box in line.get("parts", [line["box"]]):
                    pts = [(float(x) * scale, float(y) * scale) for x, y in box]
                    d.line(pts + pts[:1], fill=col, width=lw + 1)
                x, y = (float(v) * scale for v in line["box"][0])
                _tag(d, (min(x, img.width - 120), min(max(y - 24, 0), img.height - 22)),
                     "crease?" if line["kind"] == "crease" else "scratch", col, f)
            side = "F" if sc.side == "front" else "B"
            num = f"#{(index if index is not None else sc.index) + 1} "
            label = f"{num}{side} E{sc.edge_ratio:.0%} C{sc.corner_ratio:.0%}"
            if sc.specks:
                label += f" D{sc.specks}"
            _tag(d, (4, tag_y), label, SEV_COLOR[_scan_level(sc)], f)
            tag_y += fsize + 10
    return img


def annotated_photos(photos: list[bytes], est, max_side: int = 1280, only_findings: bool = True) -> list[bytes]:
    """逐張輸出標註後的大圖（Telegram 相簿用）；預設只輸出有偵測到瑕疵的"""
    ann = [a for o in (est.observations if est else []) for a in o.annotations]
    out = []
    for i, raw in enumerate(photos):
        mine = [a for a in ann if a["image"] == i]
        scans = [a["scan"] for a in mine if a["kind"] == "scan"]
        if not scans or (only_findings and not any(has_findings(s) for s in scans)):
            continue
        try:
            im = annotate_photo(raw, mine, max_size=(max_side, max_side))
        except Exception:
            continue
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
        out.append(buf.getvalue())
    return out


def _tag(d: ImageDraw.ImageDraw, xy, text: str, color: str, f) -> None:
    """白底小標籤，色塊只當識別，文字用墨色"""
    x, y = xy
    w = d.textlength(text, font=f)
    d.rectangle([x, y, x + w + 18, y + 20], fill="#ffffff", outline=color, width=1)
    d.rectangle([x + 3, y + 5, x + 11, y + 15], fill=color)
    d.text((x + 15, y + 1), text, fill=INK, font=f)


def _bar_chart(d: ImageDraw.ImageDraw, x: int, y: int, w: int, scenarios: dict) -> int:
    """水平分組長條圖；回傳畫完的 y"""
    f, fb, fs = font(17), font(19), font(15)
    d.text((x, y), "PSA grade probability", fill=INK, font=fb)
    y += 32
    # 圖例
    lx = x
    for k in ("optimistic", "base", "pessimistic"):
        d.rounded_rectangle([lx, y + 3, lx + 14, y + 17], radius=3, fill=SCEN_COLORS[k])
        d.text((lx + 20, y), SCEN_LABEL[k], fill=INK_2, font=fs)
        lx += 24 + int(d.textlength(SCEN_LABEL[k], font=fs)) + 18
    y += 30
    label_w, value_w = 70, 52
    plot_w = w - label_w - value_w
    bar_h, gap = 11, 2
    row_h = 3 * bar_h + 2 * gap + 14
    top = y
    for i in range(0, 101, 25):
        gx = x + label_w + plot_w * i / 100
        d.line([gx, top - 4, gx, top + row_h * len(GRADES)], fill=GRID, width=1)
        d.text((gx - 10, top + row_h * len(GRADES) + 2), f"{i}%", fill=INK_2, font=fs)
    for g in GRADES:
        d.text((x, y + bar_h), "PSA " + (str(g) if g != "low" else "<=4"), fill=INK, font=f)
        by = y
        for k in ("optimistic", "base", "pessimistic"):
            p = scenarios[k][g]
            bw = plot_w * p
            x0 = x + label_w
            if bw >= 1:
                d.rounded_rectangle([x0, by, x0 + bw, by + bar_h], radius=min(4, bar_h // 2), fill=SCEN_COLORS[k])
            if k == "base":  # 只在基準直接標數值
                d.text((x0 + bw + 6, by - 3), f"{p:.0%}", fill=INK, font=fs)
            by += bar_h + gap
        y += row_h
    return y + 26


def render_report(card_name: str, listing: dict, est, ev, photos: list[bytes], jpy_rate: float | None = None) -> bytes:
    W, H = 1400, 840
    img = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(img)
    f, fb, fh, fs = font(18), font(22), font(28), font(15)

    # 標頭
    verdict = ev.verdict if ev else VERDICT_SKIP
    vtext, vcol = VERDICT_STYLE.get(verdict, ("?", INK_2))
    d.rectangle([0, 0, W, 64], fill="#f0efec")
    d.rounded_rectangle([20, 14, 40, 50], radius=4, fill=vcol)
    d.text((52, 14), vtext, fill=INK, font=fh)
    title = (card_name if HAS_CJK else listing.get("id", "")) or ""
    price = listing.get("price")
    sub = f"{title}   ¥{price:,.0f}" if price else title
    if price and jpy_rate:
        sub += f" (NT${price * jpy_rate:,.0f})"
    d.text((52 + d.textlength(vtext, font=fh) + 30, 20), sub, fill=INK_2, font=fb)

    # 左：所有判讀過的照片排成格狀（最多 12 張）
    obs_list = est.observations if est else []
    ann = [a for o in obs_list for a in o.annotations]
    scanned = sorted({a["image"] for a in ann})
    show = (scanned or list(range(len(photos))))[:12]
    grid_bottom = 84
    if show:
        n = len(show)
        cols = 1 if n == 1 else 2 if n <= 4 else 3 if n <= 9 else 4
        rows = -(-n // cols)
        gap = 10
        cw = (750 - gap * (cols - 1)) // cols
        ch = min(560, (620 - gap * (rows - 1)) // rows)
        for k, idx in enumerate(show):
            r, c = divmod(k, cols)
            try:
                ph = annotate_photo(photos[idx], [a for a in ann if a["image"] == idx], max_size=(cw, ch), index=idx)
            except Exception:
                continue
            x = 20 + c * (cw + gap) + (cw - ph.width) // 2
            y0 = 84 + r * (ch + gap)
            img.paste(ph, (x, y0))
            grid_bottom = max(grid_bottom, y0 + ph.height)
    else:
        d.text((20, 100), "no photos", fill=INK_2, font=f)
    height = grid_bottom - 84
    picked = show

    # 左下：細項與照片判讀來源
    y = 84 + height + 20 if picked else 140
    merged = est.merged if est else None
    if merged and merged.subgrades:
        parts = [f"{k} {v:.1f}" for k, v in merged.subgrades.items() if v is not None]
        d.text((20, y), "Vision subgrades: " + "   ".join(parts), fill=INK, font=f)
        y += 30
    if est and est.vision_weight:
        d.text((20, y), f"Vision estimate weight in probabilities: {est.vision_weight:.0%}", fill=INK_2, font=fs)
        y += 24
    if ann:
        d.text((20, y), "F/B = front/back   E = edge whitening   C = corner whitening   D = white dots   "
                        "red circle = white spot   box = scratch / crease?", fill=INK_2, font=fs)
        y += 24
    srcs = ", ".join(sorted({o.source for o in obs_list})) or "none"
    conf = f"{merged.confidence:.0%}" if merged else "-"
    d.text((20, y), f"Photo analyzers: {srcs}   confidence used: {conf}", fill=INK_2, font=fs)

    # 右：長條圖＋摘要
    rx, rw = 800, 570
    if est and est.scenarios:
        y = _bar_chart(d, rx, 84, rw, est.scenarios)
    else:
        y = 100
    if ev:
        s = ev.scenarios

        def nt(v):
            return f"{'-' if v < 0 else '+'}NT${abs(v):,.0f}"

        rows = [
            ("Expected profit (base)", f"{nt(s['base'].ev_net)}   ROI {s['base'].roi:.0%}"),
            ("Pessimistic / optimistic", f"{nt(s['pessimistic'].ev_net)} / {nt(s['optimistic'].ev_net)}"),
            ("Grading vs selling raw", nt(ev.uplift)),
            ("PSA 10 needed to break even", f"{ev.breakeven_p10:.0%}" if ev.breakeven_p10 is not None else "never"),
            ("Max bid", (f"¥{ev.max_bid / jpy_rate:,.0f}" if jpy_rate else f"NT${ev.max_bid:,.0f}") if ev.max_bid is not None else "none"),
        ]
        for k, v in rows:
            d.text((rx, y), k, fill=INK_2, font=f)
            d.text((rx + 300, y), v, fill=INK, font=f)
            y += 30
    elif est and est.scenarios is None:
        d.text((rx, y), "Skipped: condition too poor", fill=INK, font=f)

    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()
