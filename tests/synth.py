"""測試用的合成卡片照片：圓角、背面藍色漩渦紋＋白色寶可夢球、正面黃框＋黑字，可加瑕疵、可裁特寫"""
from __future__ import annotations

import cv2
import numpy as np

W, H = 630, 880  # 63:88，約 10 px/mm


def _rounded_mask(w: int, h: int, r: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(m, (0, r), (w, h - r), 255, -1)
    for cx, cy in [(r, r), (w - r, r), (r, h - r), (w - r, h - r)]:
        cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def back_card(rng: np.random.Generator) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    swirl = (np.sin(xx / 37 + np.cos(yy / 53) * 2) * 25).astype(np.int16)
    img[..., 0] = np.clip(170 + swirl, 0, 255)  # BGR：藍
    img[..., 1] = np.clip(70 + swirl // 2, 0, 255)
    img[..., 2] = 25
    cv2.circle(img, (W // 2, H // 2), 150, (40, 40, 40), 14)
    cv2.ellipse(img, (W // 2, H // 2), (143, 143), 0, 0, 180, (245, 245, 245), -1)   # 球下半白色
    cv2.ellipse(img, (W // 2, H // 2), (143, 143), 0, 180, 360, (40, 40, 210), -1)   # 上半紅色
    cv2.circle(img, (W // 2, H // 2), 40, (245, 245, 245), -1)
    cv2.putText(img, "POKEMON", (110, 150), cv2.FONT_HERSHEY_DUPLEX, 2.2, (30, 200, 250), 6)
    return img


def front_card(rng: np.random.Generator, left: int = 40, right: int = 40, top: int = 45, bottom: int = 45) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (30, 205, 240)  # 黃框
    art = img[top:H - bottom, left:W - right]
    art[:] = (150, 110, 90)
    cv2.circle(art, (art.shape[1] // 2, art.shape[0] // 3), 120, (220, 220, 230), -1)  # 圖面裡的白色
    cv2.putText(img, "Illus. Ken Sugimori  047/P", (left + 10, H - bottom // 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    return img


def chip_edges(img: np.ndarray, rng: np.random.Generator, n: int = 25, size: int = 6) -> None:
    """沿邊緣點白點（白邊）"""
    for _ in range(n):
        side = rng.integers(4)
        t = rng.uniform(0.15, 0.85)
        x, y = {0: (int(t * W), 0), 1: (int(t * W), H - 1), 2: (0, int(t * H)), 3: (W - 1, int(t * H))}[side]
        cv2.circle(img, (x, y), int(rng.integers(size // 2, size + 1)), (240, 240, 240), -1)


def white_corner(img: np.ndarray, r: int = 18) -> None:
    for cx, cy in [(0, 0), (W, 0), (0, H), (W, H)]:
        cv2.circle(img, (cx, cy), r + 12, (240, 240, 240), -1)


def crease(img: np.ndarray) -> None:
    cv2.line(img, (60, 300), (580, 520), (225, 230, 235), 2)


def scratch(img: np.ndarray) -> None:
    cv2.line(img, (420, 650), (480, 700), (225, 230, 235), 2)


def photo(card: np.ndarray, bg=(45, 45, 45), margin: int = 90, angle: float = 0.0, seed: int = 0) -> np.ndarray:
    """把卡片放到背景上（圓角、可微轉）、加雜訊"""
    rng = np.random.default_rng(seed)
    mask = _rounded_mask(W, H, 30)
    out = np.full((H + 2 * margin, W + 2 * margin, 3), bg, np.uint8)
    region = out[margin:margin + H, margin:margin + W]
    region[mask > 0] = card[mask > 0]
    if angle:
        m = cv2.getRotationMatrix2D((out.shape[1] / 2, out.shape[0] / 2), angle, 1.0)
        out = cv2.warpAffine(out, m, (out.shape[1], out.shape[0]), borderValue=bg)
    noise = rng.normal(0, 4, out.shape)
    return np.clip(out + noise, 0, 255).astype(np.uint8)


def closeup(img: np.ndarray, corner: str, frac: float = 0.4, zoom: float = 2.5) -> np.ndarray:
    """裁出四角特寫並放大（模擬賣家拍的角落放大圖）"""
    h, w = img.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    y0 = 0 if corner[0] == "t" else h - ch
    x0 = 0 if corner[1] == "l" else w - cw
    crop = img[y0:y0 + ch, x0:x0 + cw]
    return cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)


def jpeg(img: np.ndarray, q: int = 85) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()
