"""本機逐張檢查卡況（OpenCV，不呼叫任何 API）

賣家通常會拍正反面全圖再加四角、邊緣的放大特寫。這裡每張照片都跑一次：

1. 判斷正面（黃框＋圖）、新版背面（黃框＋藍底，上下各一個黃色 POKéMON）、
   旧裏背面（藍底為主）；都不是的照片（包裝、說明圖）略過。
2. 用卡框（黃框或藍底）的顏色區塊取凸包，當成卡片的輪廓。
3. 白邊：沿凸包外緣每 2 px 取一條垂直邊緣的短線，線上有「白」就算這一點白掉。
   「白」= 絕對偏白，或比同一點內側的卡框明顯更淡更亮（照片壓縮後的細白邊）。
   卡套、硬殼（外圍灰色）時只看內側；白背景跟白邊分不開，該張不判斷邊緣並註記。
   靠近四角的點另外算「轉角」；被照片邊界切掉的那一側不算。
4. 白點：黃框卡只在卡框那一圈算（圖面、logo 亮部太容易誤判）；旧裏背面看整面。
5. 刮痕／摺痕：在卡面顏色內，用形態學 closing 找「被顏色包圍的細白線」：
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
WHITE_S, WHITE_V = 60, 170      # 白：低飽和、高亮度
REL_S, REL_V = 40, 35           # 或比卡框中位色淡 40、亮 35 以上（照片壓縮後的白邊）
LIGHT_V = 140                   # 卡片外圍中位亮度高於此（且低飽和）= 白背景／卡套
LOC_S, LOC_V = 50, 25           # 跟同一點內側卡框比：淡 50、亮 25 以上（一般邊緣反光只會淡一點、不會更亮）
LOC_ON = True
EDGE_RUN = 1                    # 沿輪廓連續 ≥ 這麼多取樣點白掉才算白邊（校準時 4 以上會把真白邊也濾掉，先關）
EDGE_IN = 0.006                 # 白邊取樣線往卡內的深度（短邊比例）
EDGE_MARGIN = 0.003             # 背景定位成功時，取樣線避開輪廓線這麼深（卡邊混色像素）
FRAME = 0.08                    # 黃框卡的卡框寬約短邊 6~8%，白點只在這圈裡算

SEV_EDGE = (0.02, 0.06)         # 邊緣取樣點白掉比例：< 2% 無、< 6% 輕微、以上明顯
SEV_CORNER = (0.06, 0.20)       # 轉角本來就容易被反光干擾，門檻放寬
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
    """黃色包住整張卡 = 黃框；框內藍色為主 = 新版背面（e 卡以後），否則正面。
    藍色為主、黃色只是小塊（旧裏背面）= 背面"""
    blue_m, yellow_m = cv2.inRange(hsv, *BLUE), cv2.inRange(hsv, *YELLOW)
    blue, yellow = blue_m.mean() / 255, yellow_m.mean() / 255
    if yellow > 0.02:
        ya, ba = _hull_area(yellow_m), _hull_area(blue_m)
        if ya >= 0.6 * max(ya, ba):
            return "back" if _framed_blue(yellow_m, blue_m) else "front"
    if blue > 0.15:
        return "back"
    if yellow > 0.04:
        return "front"
    return None


def _framed_blue(yellow_m: np.ndarray, blue_m: np.ndarray) -> bool:
    """新版背面：黃框裡上下各有一個黃色 POKéMON 字樣、中間（寶可夢球）沒有黃色、四周是藍色漩渦。
    正面的圖或文字框會讓中間那段也有黃色，或上下沒有黃色。"""
    pts = cv2.findNonZero(_main_region(yellow_m))
    if pts is None:
        return False
    x, y, w, h = cv2.boundingRect(cv2.convexHull(pts))
    x0, x1 = x + int(w * 0.12), x + w - int(w * 0.12)

    def frac(m, t0, t1):
        sub = m[y + int(h * t0): y + int(h * t1), x0:x1]
        return sub.mean() / 255 if sub.size else 0.0

    return (frac(yellow_m, 0.15, 0.35) > 0.12 and frac(yellow_m, 0.65, 0.85) > 0.12
            and frac(yellow_m, 0.42, 0.58) < 0.08 and frac(blue_m, 0.1, 0.9) > 0.1)


def _main_region(mask: np.ndarray) -> np.ndarray:
    """去雜點，只留面積夠大的色塊"""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a >= areas.max() * 0.02]
    return np.isin(labels, keep).astype(np.uint8) * 255


BG_AB, BG_L, BG_DARK = 14, 45, 80  # 跟背景色比：色度差 ≥ 14、亮 45 以上、或暗 80 以上才算不是背景（陰影只會暗一些）
BG_FILL = 0.8                    # 前景面積 / 凸包面積 ≥ 0.8 才相信這個凸包是卡片


def locate_card(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """用照片四邊估背景色（桌面可能不只一色，分 2 群），離背景色夠遠的最大色塊當卡片。
    比只靠黃框／藍底的顏色穩：木桌跟黃框顏色很像、旧裏背面的淺色邊不在藍色裡，
    特寫被切到邊的卡也只會取到看得見的部分。回傳 (卡片遮罩, 凸包)。"""
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int32)
    b = max(4, int(min(h, w) * 0.03))
    ring = np.zeros((h, w), bool)
    ring[:b, :] = ring[-b:, :] = ring[:, :b] = ring[:, -b:] = True
    data = lab[ring].astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lbl, centers = cv2.kmeans(data, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(lbl.ravel(), minlength=2)
    fg = np.ones((h, w), bool)
    for c, n in zip(centers.astype(np.int32), counts):
        if n < 0.15 * len(data):
            continue  # 太少的那群大概是卡片本身被切到邊
        dab = np.sqrt(((lab[..., 1:] - c[1:]) ** 2).sum(axis=2))
        dl = lab[..., 0] - c[0]
        fg &= (dab > BG_AB) | (dl > BG_L) | (dl < -BG_DARK)
    fg = fg.astype(np.uint8) * 255
    k = max(3, int(min(h, w) * 0.01)) | 1
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3 * k, 3 * k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < 0.1 * h * w:
        return None
    m = (labels == i).astype(np.uint8) * 255
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    m |= cv2.bitwise_not(ff)  # 填掉反光造成的洞
    hull = cv2.convexHull(cv2.findNonZero(m))
    if cv2.countNonZero(m) < BG_FILL * cv2.contourArea(hull):
        return None  # 前景形狀跟凸包差太多：陰影、桌面雜物混進來了
    hull_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(hull_mask, [hull], 255)
    return hull_mask, hull


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

    blue_m, yellow_m = cv2.inRange(hsv, *BLUE), cv2.inRange(hsv, *YELLOW)
    framed = side == "front" or _framed_blue(yellow_m, blue_m)  # 黃框卡（正面或新版背面）
    color = _main_region(yellow_m if framed else blue_m)
    # 卡面圖案的顏色（找刮痕用）：新版背面 = 黃框＋藍底
    surface = _main_region(cv2.bitwise_or(yellow_m, blue_m)) if side == "back" and framed else color
    pts = cv2.findNonZero(color)
    if pts is None or len(pts) < 100:
        res.side = None
        return res
    loc = locate_card(bgr)
    if loc is not None:
        hull_mask, hull = loc
        pts = cv2.findNonZero(hull_mask)
    else:
        hull = cv2.convexHull(pts)
        hull_mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(hull_mask, [hull], 255)
        res.notes.append(f"#{index + 1}：背景分不出來，改用卡框顏色估卡片範圍")
    x, y, bw, bh = cv2.boundingRect(hull)
    short = min(bw, bh)
    res.hull = (hull.astype(np.float32) * scale).astype(np.int32)
    res.card_px = int(cv2.countNonZero(hull_mask) * scale * scale)

    k = int(np.clip(short * 0.012, 3, 25))
    # 被照片邊界切掉的那幾側不算邊緣
    border = np.zeros((h, w), np.uint8)
    border[: k * 2, :] = border[-k * 2:, :] = 255
    border[:, : k * 2] = border[:, -k * 2:] = 255

    s, v = hsv[..., 1], hsv[..., 2]
    whiteish = (s < WHITE_S) & (v > WHITE_V)
    # 白邊常常只是比卡框亮、淡一截（照片壓縮後不會是純白），所以也用卡框本身的顏色當基準比
    inner = cv2.erode(hull_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    band = cv2.bitwise_and(cv2.subtract(hull_mask, inner), cv2.bitwise_not(border))
    ref = hsv[(band > 0) & ((color > 0) if loc is None else (hull_mask > 0))]
    if len(ref) > 50:
        ref_s, ref_v = float(np.median(ref[:, 1])), float(np.median(ref[:, 2]))
        relwhite = (v >= ref_v + REL_V) & (s <= ref_s - REL_S)
    else:
        relwhite = np.zeros_like(whiteish)
    whitening = whiteish | relwhite

    # 只看卡片外面緊鄰的一圈：卡套、硬殼、白桌面都會讓這圈偏白
    ring = cv2.subtract(cv2.dilate(hull_mask, np.ones((k * 8 + 1, k * 8 + 1), np.uint8)),
                        cv2.dilate(hull_mask, np.ones((k * 3 + 1, k * 3 + 1), np.uint8)))
    bg = hsv[(ring > 0) & (border == 0)]
    bg_white = len(bg) > 50 and np.median(bg[:, 1]) < WHITE_S and np.median(bg[:, 2]) > LIGHT_V
    # 卡套／硬殼：外圍是低飽和的灰（壓克力、塑膠），卡邊會被打亮，也只看內側
    cased = not bg_white and len(bg) > 50 and np.median(bg[:, 1]) < 35 and np.median(bg[:, 2]) > 85
    if cased:
        res.notes.append(f"#{index + 1}：卡片可能在卡套或硬殼裡，邊緣只看內側")
    bg_white = bg_white or cased
    white_bg_only = bg_white and not cased
    if white_bg_only:
        # 白背景跟白邊分不開（卡邊會跟背景混色），這張不判斷邊緣
        res.notes.append(f"#{index + 1}：背景偏白，白邊無法判斷")

    # 轉角：色塊最小外接矩形的四個角（沒碰到照片邊界的才算）
    rect = cv2.boxPoints(cv2.minAreaRect(pts))
    corner_zone = np.zeros((h, w), np.uint8)
    zone_r = max(k * 5, int(short * 0.1))  # 卡片圓角約占寬度 5%，特寫時圓角更大
    for cx, cy in rect:
        if k < cx < w - k and k < cy < h - k:
            cv2.circle(corner_zone, (int(cx), int(cy)), zone_r, 255, -1)

    # 沿凸包外緣每 2 px 取一條垂直邊緣的短線（往內 r_in、往外 r_out），線上有白就算這一點白掉。
    # 邊緣白掉時卡片顏色會缺一塊，凸包仍沿著原本的邊走，所以白掉的部分會落在這條短線上。
    near_color = cv2.dilate(color, np.ones((k * 3 + 1, k * 3 + 1), np.uint8)) if loc is None else hull_mask
    r_in = max(3, int(short * EDGE_IN))
    r_out = 0 if bg_white or loc is not None else max(3, int(short * 0.012))
    # 只看內側時避開輪廓線上那幾格（卡邊跟背景混色的像素、圓角處凸包弦線會稍微切到背景）
    margin = max(2, int(short * EDGE_MARGIN)) if loc is not None else 2
    offs = np.arange(-r_in, (r_out if r_out else -margin) + 1, dtype=np.float32)
    poly = hull.reshape(-1, 2).astype(np.float32)
    center = poly.mean(axis=0)
    edge_white = np.zeros((h, w), np.uint8)
    flags = {"edge": [], "corner": []}
    samples = []  # 沿輪廓依序：(白掉?, 區域, 白掉像素 y, x)
    for i in range(0 if white_bg_only else len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        seg = float(np.linalg.norm(b - a))
        if seg < 1:
            continue
        d = (b - a) / seg
        nrm = np.array([d[1], -d[0]], np.float32)
        if np.dot(nrm, (a + b) / 2 - center) < 0:
            nrm = -nrm
        for t in np.arange(0, seg, 2.0, dtype=np.float32):
            p = a + d * t
            px, py = int(p[0]), int(p[1])
            if not (k <= px < w - k and k <= py < h - k) or not near_color[min(h - 1, max(0, py)), min(w - 1, max(0, px))]:
                continue  # 卡片被照片邊界切到，或這段凸包斜切過圖面（正面特寫的 L 形黃框）
            xs = np.rint(p[0] + nrm[0] * offs).astype(int)
            ys = np.rint(p[1] + nrm[1] * offs).astype(int)
            keep = (xs >= 0) & (ys >= 0) & (xs < w) & (ys < h)
            xs, ys = xs[keep], ys[keep]
            fx, fy = int(p[0] + nrm[0] * (r_out + 2 * k)), int(p[1] + nrm[1] * (r_out + 2 * k))
            if r_out and 0 <= fx < w and 0 <= fy < h and whitening[fy, fx]:
                continue  # 外面也是白的（卡套、壓克力殼、亮背景），分不出是不是白邊
            hit = whitening[ys, xs].copy()
            # 跟同一點往內一點的卡框比：明顯更淡、更亮也算（壓縮過的細白邊）
            ix = np.rint(p[0] - nrm[0] * np.arange(r_in + k, r_in + 2 * k)).astype(int)
            iy = np.rint(p[1] - nrm[1] * np.arange(r_in + k, r_in + 2 * k)).astype(int)
            if ix.min() >= 0 and iy.min() >= 0 and ix.max() < w and iy.max() < h and hull_mask[iy, ix].all():
                ls, lv = np.median(s[iy, ix]), np.median(v[iy, ix])
                seg_s, seg_v = s[ys, xs].astype(int), v[ys, xs].astype(int)
                if LOC_ON:
                    hit |= (seg_s <= ls - LOC_S) & (seg_v >= lv + LOC_V) & (seg_v > 90)
            samples.append((bool(hit.any()), "corner" if corner_zone[py, px] else "edge", ys[hit], xs[hit]))

    # 白邊是連續一段，零星一兩個取樣點白掉多半是反光、圖案邊：只算連續 ≥ EDGE_RUN 點的段落
    hits = np.array([sm[0] for sm in samples], bool)
    keep = np.zeros_like(hits)
    i = 0
    while i < len(hits):
        if hits[i]:
            j = i
            while j < len(hits) and hits[j]:
                j += 1
            if j - i >= EDGE_RUN:
                keep[i:j] = True
            i = j
        else:
            i += 1
    for ok, (_, zone, hy, hx) in zip(keep, samples):
        flags[zone].append(bool(ok))
        if ok:
            edge_white[hy, hx] = 255

    # 角尖整個白掉時，那一大塊白會讓上面的「外面也是白」判斷跳過，所以另外看
    # 「最小外接矩形的角落、凸包外」那一小塊：背景不是白色時，那裡出現白色就是角尖白掉
    rect_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(rect_mask, [rect.astype(np.int32)], 255)
    tip = cv2.bitwise_and(cv2.bitwise_and(rect_mask, cv2.bitwise_not(hull_mask)), corner_zone)
    tip = cv2.bitwise_and(tip, cv2.bitwise_not(border))
    tip_ratio = 0.0
    if not bg_white and cv2.countNonZero(tip):
        tip_white = cv2.bitwise_and(tip, whiteish.astype(np.uint8) * 255)
        tip_ratio = cv2.countNonZero(tip_white) / cv2.countNonZero(tip)
        edge_white = cv2.bitwise_or(edge_white, tip_white)

    res.edge_ratio = float(np.mean(flags["edge"])) if flags["edge"] else 0.0
    res.corner_ratio = max(float(np.mean(flags["corner"])) if flags["corner"] else 0.0, tip_ratio)
    res.edge_severity = _severity(res.edge_ratio, SEV_EDGE)
    res.corner_severity = _severity(res.corner_ratio, SEV_CORNER)
    whiteish = whiteish.astype(np.uint8) * 255

    # 細白線（刮痕、摺痕）與白點：被卡片顏色包圍的細小非色塊
    kk = int(np.clip(short * 0.015, 5, 31)) | 1
    closed = cv2.morphologyEx(surface, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
    interior = cv2.erode(hull_mask, np.ones((4 * k + 1, 4 * k + 1), np.uint8))
    thin = cv2.bitwise_and(cv2.bitwise_and(cv2.subtract(closed, surface), whiteish), interior)
    # 黃框卡的白點只算卡框那一圈：圖面、logo 的亮部太容易誤判
    frame = cv2.subtract(hull_mask, cv2.erode(hull_mask, np.ones((int(short * FRAME) | 1,) * 2, np.uint8))) if framed else hull_mask
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
        is_line = length >= short * SCRATCH_LEN and length / width >= 4
        if not is_line and not frame[int(cy), int(cx)]:
            continue
        defect[comp > 0] = 255
        if is_line:
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
