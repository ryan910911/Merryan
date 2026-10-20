"""本機逐張檢查卡況（OpenCV，不呼叫任何 API）

賣家通常會拍正反面全圖再加四角、邊緣的放大特寫。這裡每張照片都跑一次：

1. 判斷是正面（黃框）還是背面（藍底）；兩者都不是的照片（包裝、說明圖）略過。
2. 用黃框／藍底的顏色區塊取凸包，當成「卡片應該有顏色的範圍」。
   凸包會跨過邊緣上的小缺口，所以邊緣白掉的地方會落在凸包內、卻不是卡片顏色。
3. 白邊／白點：凸包內緣一圈（band）裡偏白的像素比例；靠近凸包轉角的另外算「四角」。
   被照片邊界切掉的那一側（特寫照常見）不算。
4. 刮痕／摺痕：在藍底或黃框內，用形態學 closing 找「被顏色包圍的細白線」：
   大面積的白色圖案（寶可夢球、文字底）不會被 closing 填滿，所以不會誤判。
   夠長的細線當刮痕；橫跨卡片一大段的直線當疑似摺痕。

這是看顏色的規則判斷：卡套反光、壓膜、白色背景沾到邊會造成誤判，
所以報告圖會把偵測到的位置標出來，讓人用眼睛確認。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MAX_SIDE = 1600

# OpenCV HSV：H 0~180、S/V 0~255
BLUE = ((95, 80, 50), (130, 255, 255))
YELLOW = ((16, 90, 120), (36, 255, 255))
WHITE_S, WHITE_V = 60, 170      # 白邊：低飽和、高亮度
LINE_S, LINE_V = 90, 140        # 細線：比白邊寬鬆一點

SEV_EDGE = (0.01, 0.04)         # band 白像素比例：< 1% 無、< 4% 輕微、以上明顯
SEV_CORNER = (0.03, 0.10)       # 轉角本來就容易被反光干擾，門檻放寬
SCRATCH_LEN = 0.06              # 細線長度 ≥ 可見卡片短邊 6% 算刮痕
CREASE_LEN = 0.30               # ≥ 30% 且夠直，算疑似摺痕


@dataclass
class PhotoScan:
    index: int
    side: str | None  # "front" / "back" / None
    width: int = 0
    height: int = 0
    edge_ratio: float = 0.0
    corner_ratio: float = 0.0
    edge_severity: str = "none"
    corner_severity: str = "none"
    lines: list = field(default_factory=list)  # [{"len": 比例, "box": 4 點, "kind": "scratch"/"crease"}]
    specks: int = 0
    hull: np.ndarray | None = None  # 凸包輪廓（原圖座標）
    defect_mask: np.ndarray | None = None  # 白邊／白點像素（處理尺寸）
    scale: float = 1.0  # 處理尺寸 -> 原圖
    card_px: int = 0  # 可見卡片面積（像素），解析度參考
    notes: list = field(default_factory=list)

    @property
    def crease_suspect(self) -> bool:
        return any(l["kind"] == "crease" for l in self.lines)


def _severity(ratio: float, cuts: tuple) -> str:
    return "none" if ratio < cuts[0] else "light" if ratio < cuts[1] else "heavy"


def decode(raw: bytes) -> tuple[np.ndarray, float] | None:
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return None
    h, w = arr.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    if s < 1:
        arr = cv2.resize(arr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return arr, 1 / s


def _hull_area(mask: np.ndarray) -> float:
    pts = cv2.findNonZero(_main_region(mask))
    return cv2.contourArea(cv2.convexHull(pts)) if pts is not None and len(pts) >= 3 else 0.0


def classify(hsv: np.ndarray) -> str | None:
    """黃色包住整張卡（黃色凸包跟藍色一樣大）= 正面；藍色為主、黃色只是小塊（背面 logo）= 背面"""
    blue_m, yellow_m = cv2.inRange(hsv, *BLUE), cv2.inRange(hsv, *YELLOW)
    blue, yellow = blue_m.mean() / 255, yellow_m.mean() / 255
    if yellow > 0.02:
        ya, ba = _hull_area(yellow_m), _hull_area(blue_m)
        if ya >= 0.6 * max(ya, ba):
            return "front"
    if blue > 0.15:
        return "back"
    if yellow > 0.04:
        return "front"
    return None


def _main_region(mask: np.ndarray) -> np.ndarray:
    """去雜點，只留面積夠大的色塊"""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a >= areas.max() * 0.02]
    return np.isin(labels, keep).astype(np.uint8) * 255


def scan_photo(raw: bytes, index: int = 0) -> PhotoScan:
    dec = decode(raw)
    if dec is None:
        return PhotoScan(index, None, notes=["無法讀取圖片"])
    bgr, scale = dec
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    side = classify(hsv)
    res = PhotoScan(index, side, w, h, scale=scale)
    if side is None:
        return res

    color = _main_region(cv2.inRange(hsv, *(BLUE if side == "back" else YELLOW)))
    pts = cv2.findNonZero(color)
    if pts is None or len(pts) < 100:
        res.side = None
        return res
    hull = cv2.convexHull(pts)
    hull_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(hull_mask, [hull], 255)
    x, y, bw, bh = cv2.boundingRect(hull)
    short = min(bw, bh)
    res.hull = (hull.astype(np.float32) * scale).astype(np.int32)
    res.card_px = int(cv2.countNonZero(hull_mask) * scale * scale)

    k = int(np.clip(short * 0.012, 3, 25))
    inner = cv2.erode(hull_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    band = cv2.subtract(hull_mask, inner)
    # 被照片邊界切掉的那幾側不算邊緣
    border = np.zeros((h, w), np.uint8)
    border[: k * 2, :] = border[-k * 2:, :] = 255
    border[:, : k * 2] = border[:, -k * 2:] = 255
    band = cv2.bitwise_and(band, cv2.bitwise_not(border))
    # 只看貼著卡片顏色的那段凸包邊緣：正面特寫的黃框是 L 形，凸包會斜切過圖面，那段不是卡片邊
    band = cv2.bitwise_and(band, cv2.dilate(color, np.ones((k * 3 + 1, k * 3 + 1), np.uint8)))

    s, v = hsv[..., 1], hsv[..., 2]
    whiteish = ((s < WHITE_S) & (v > WHITE_V)).astype(np.uint8) * 255
    edge_white = cv2.bitwise_and(cv2.bitwise_and(band, whiteish), cv2.bitwise_not(color))

    # 轉角：用色塊的最小外接矩形取四個角（沒碰到照片邊界的才算）。
    # 角尖整個白掉時凸包會把它切掉，所以另外看「矩形角落、凸包外」那一小塊：
    # 背景不是白色時，那裡出現白色就是角尖白掉；背景偏白時無法分辨，略過並記下。
    rect = cv2.boxPoints(cv2.minAreaRect(pts))
    corner_zone = np.zeros((h, w), np.uint8)
    zone_r = max(k * 5, int(short * 0.1))  # 卡片圓角約占寬度 5%，特寫時圓角更大
    for cx, cy in rect:
        if k * 3 < cx < w - k * 3 and k * 3 < cy < h - k * 3:
            cv2.circle(corner_zone, (int(cx), int(cy)), zone_r, 255, -1)
    outside = cv2.dilate(hull_mask, np.ones((k * 4 + 1, k * 4 + 1), np.uint8))
    bg = hsv[(outside == 0) & (border == 0)]
    bg_white = len(bg) > 50 and np.median(bg[:, 1]) < WHITE_S and np.median(bg[:, 2]) > WHITE_V
    rect_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(rect_mask, [rect.astype(np.int32)], 255)
    tip = cv2.bitwise_and(cv2.bitwise_and(rect_mask, cv2.bitwise_not(hull_mask)), corner_zone)
    tip = cv2.bitwise_and(tip, cv2.bitwise_not(border))
    if bg_white:
        res.notes.append(f"#{index + 1}：背景偏白，角尖白掉可能看不出來")
        tip[:] = 0
    tip_white = cv2.bitwise_and(tip, whiteish)
    edge_white = cv2.bitwise_or(edge_white, tip_white)

    corner_band = cv2.bitwise_and(band, corner_zone)
    edge_band = cv2.bitwise_and(band, cv2.bitwise_not(corner_zone))
    if cv2.countNonZero(edge_band):
        res.edge_ratio = cv2.countNonZero(cv2.bitwise_and(edge_white, edge_band)) / cv2.countNonZero(edge_band)
    corner_total = cv2.countNonZero(corner_band) + cv2.countNonZero(tip)
    if corner_total:
        res.corner_ratio = (cv2.countNonZero(cv2.bitwise_and(edge_white, corner_band)) + cv2.countNonZero(tip_white)) / corner_total
    res.edge_severity = _severity(res.edge_ratio, SEV_EDGE)
    res.corner_severity = _severity(res.corner_ratio, SEV_CORNER)

    # 細白線（刮痕、摺痕）與白點：被卡片顏色包圍的細小非色塊
    kk = int(np.clip(short * 0.015, 5, 31)) | 1
    closed = cv2.morphologyEx(color, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
    lightish = ((s < LINE_S) & (v > LINE_V)).astype(np.uint8) * 255
    interior = cv2.erode(hull_mask, np.ones((4 * k + 1, 4 * k + 1), np.uint8))
    thin = cv2.bitwise_and(cv2.bitwise_and(cv2.subtract(closed, color), lightish), interior)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(thin, 8)
    defect = edge_white.copy()
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 4:
            continue
        comp = (labels == i).astype(np.uint8)
        cpts = cv2.findNonZero(comp)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(cpts)
        length = max(rw, rh)
        width = max(1.0, min(rw, rh))
        defect[comp > 0] = 255
        if length >= short * SCRATCH_LEN and length / width >= 4:
            kind = "crease" if length >= short * CREASE_LEN and length / width >= 12 else "scratch"
            box = cv2.boxPoints(((cx, cy), (rw, rh), ang)) * scale
            res.lines.append({"len": length / short, "box": box.astype(np.int32), "kind": kind})
        else:
            res.specks += 1
    res.lines = _merge_lines(res.lines, short, scale)
    res.defect_mask = defect

    zh = "背面" if side == "back" else "正面"
    if res.edge_severity != "none" or res.corner_severity != "none":
        res.notes.append(f"#{index + 1} {zh}：邊緣白 {res.edge_ratio:.1%}、轉角白 {res.corner_ratio:.1%}")
    for l in res.lines:
        res.notes.append(f"#{index + 1} {zh}：{'疑似摺痕' if l['kind'] == 'crease' else '細刮痕'}（長約卡片短邊 {l['len']:.0%}）")
    if res.specks >= 3:
        res.notes.append(f"#{index + 1} {zh}：白點 {res.specks} 處")
    return res


def _merge_lines(lines: list, short: float, scale: float) -> list:
    """同一條摺痕常被圖案切成幾段：方向相近、在同一直線上的段落合併計算長度"""
    if len(lines) < 2:
        return lines
    segs = []
    for l in lines:
        pts = l["box"].astype(np.float32) / scale
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        if vx < 0:
            vx, vy = -vx, -vy
        segs.append((l, np.arctan2(vy, vx), (x0, y0), (vx, vy)))
    used = set()
    out = []
    for i, (li, ai, pi, di) in enumerate(segs):
        if i in used:
            continue
        group = [li]
        used.add(i)
        for j in range(i + 1, len(segs)):
            lj, aj, pj, _ = segs[j]
            if j in used or abs(ai - aj) > np.radians(4):
                continue
            # pj 到 i 直線的垂直距離
            dist = abs((pj[0] - pi[0]) * di[1] - (pj[1] - pi[1]) * di[0])
            if dist < short * 0.015:
                group.append(lj)
                used.add(j)
        total = sum(g["len"] for g in group)
        kind = "crease" if total >= CREASE_LEN or any(g["kind"] == "crease" for g in group) else "scratch"
        merged = dict(group[0], len=total, kind=kind, parts=[g["box"] for g in group])
        out.append(merged)
    return out


def worst(levels) -> str | None:
    order = {"none": 0, "light": 1, "heavy": 2}
    levels = [l for l in levels if l]
    return max(levels, key=order.get) if levels else None
