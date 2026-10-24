"""用標註頁的人工標記校準 defect_scan：算漏抓／誤抓，並掃門檻

用法：python tools/calibrate.py LABELS_DIR [--grid]
  LABELS_DIR/photos/<id>.json  標註頁匯出的文件（ArtifactData list --out_dir）
  LABELS_DIR/img/<asset>.jpg   對應的照片（Artifact read path=<asset>）

比對方式（人工標記是正規化座標的框／多邊形）：
- 命中：人工框內有偵測到的瑕疵像素 → 抓到；沒有 → 漏抓
- 誤抓：偵測到的瑕疵連通區塊不在任何人工框（外擴一點）內
- 只比白邊／角／白點／刮痕這些 defect_scan 看得到的類型；壓痕、髒污不算漏抓
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import defect_scan  # noqa: E402

VISIBLE = {"edge", "corner", "spot", "scratch", "crease"}  # defect_scan 理論上能看到的
PAD = 0.02  # 人工框外擴（短邊比例），畫圈不會剛好貼著瑕疵


def load(labels: Path) -> list[dict]:
    out = []
    for p in sorted((labels / "photos").glob("*.json")):
        d = json.loads(p.read_text())
        if not d.get("done") or not d.get("asset"):
            continue
        img = labels / "img" / f"{d['asset']}.jpg"
        if not img.exists():
            continue
        d["id"], d["path"] = p.stem, img
        out.append(d)
    return out


def mark_mask(marks: list[dict], w: int, h: int, pad: int) -> tuple[np.ndarray, list]:
    """人工標記 → 像素遮罩（處理尺寸），以及每個標記的 (type, mask)"""
    allm = np.zeros((h, w), np.uint8)
    per = []
    for m in marks:
        if m["t"] not in VISIBLE:
            continue
        mk = np.zeros((h, w), np.uint8)
        if m.get("pts") and len(m["pts"]) > 2:
            poly = np.array([[x * w, y * h] for x, y in m["pts"]], np.int32)
            cv2.fillPoly(mk, [poly], 255)
        else:
            x0, y0, x1, y1 = m["box"]
            cv2.rectangle(mk, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), 255, -1)
        if pad:
            mk = cv2.dilate(mk, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))
        per.append((m["t"], mk))
        allm |= mk
    return allm, per


def evaluate(photos: list[dict], verbose: bool = False) -> dict:
    tp = miss = fp = 0
    by_type: dict[str, list[int]] = {}
    photo_rows = []
    for d in photos:
        raw = d["path"].read_bytes()
        res = defect_scan.scan_photo(raw)
        if res.side is None or res.defect_mask is None:
            # 沒認出卡片：人工有標就全算漏抓
            n = sum(1 for m in d["marks"] if m["t"] in VISIBLE)
            miss += n
            photo_rows.append((d["id"], d["group"], "no-card", n, 0, 0))
            continue
        det = res.defect_mask.copy()
        h, w = det.shape
        # 刮痕／摺痕的框也算偵測到的區域
        for l in res.lines:
            box = (np.array(l["box"], np.float32) / res.scale).astype(np.int32)
            cv2.fillPoly(det, [box], 255)
        short = min(w, h)
        gt, per = mark_mask(d["marks"], w, h, int(short * PAD))
        hit = 0
        for t, mk in per:
            ok = bool(cv2.countNonZero(cv2.bitwise_and(det, mk)))
            by_type.setdefault(t, [0, 0])[0 if ok else 1] += 1
            hit += ok
        miss_here = len(per) - hit
        # 誤抓：偵測區塊不在任何人工框內。小碎片合併後再數
        det_blob = cv2.dilate(det, np.ones((int(short * 0.01) | 1,) * 2, np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(det_blob, 8)
        fp_here = 0
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 12:
                continue
            comp = (labels == i).astype(np.uint8) * 255
            if not cv2.countNonZero(cv2.bitwise_and(comp, gt)):
                fp_here += 1
        tp += hit
        miss += miss_here
        fp += fp_here
        photo_rows.append((d["id"], d["group"], res.side, miss_here, hit, fp_here,
                           f"edge {res.edge_ratio:.1%}/{res.edge_severity} corner {res.corner_ratio:.1%}/{res.corner_severity} "
                           f"lines {len(res.lines)} specks {res.specks}"))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + miss) if tp + miss else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    if verbose:
        for r in photo_rows:
            print(" ", *r)
        for t, (ok, no) in sorted(by_type.items()):
            print(f"  {t}: 抓到 {ok} / 漏 {no}")
    return {"tp": tp, "miss": miss, "fp": fp, "precision": prec, "recall": rec, "f1": f1, "by_type": by_type}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", type=Path)
    ap.add_argument("--grid", action="store_true", help="掃門檻組合")
    a = ap.parse_args()
    photos = load(a.labels)
    print(f"{len(photos)} 張已標照片，{sum(len(d['marks']) for d in photos)} 個標記")
    base = evaluate(photos, verbose=True)
    print("目前設定：", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in base.items() if k != "by_type"})
    if not a.grid:
        return
    space = {
        "WHITE_V": [150, 170, 190],
        "REL_V": [25, 35, 50],
        "LOC_V": [15, 25, 40],
        "LOC_S": [35, 50, 70],
        "SCRATCH_LEN": [0.04, 0.06, 0.09],
    }
    keys = list(space)
    orig = {k: getattr(defect_scan, k) for k in keys}
    results = []
    for combo in itertools.product(*space.values()):
        for k, v in zip(keys, combo):
            setattr(defect_scan, k, v)
        r = evaluate(photos)
        results.append((r["f1"], r["recall"], r["precision"], dict(zip(keys, combo))))
    for k, v in orig.items():
        setattr(defect_scan, k, v)
    results.sort(key=lambda x: (-x[0], -x[1]))
    print("\n最佳 10 組：")
    for f1, rec, prec, cfg in results[:10]:
        print(f"  F1 {f1:.3f} recall {rec:.3f} precision {prec:.3f}  {cfg}")


if __name__ == "__main__":
    main()
