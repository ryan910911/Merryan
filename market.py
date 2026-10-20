"""查 Mercari 近期成交價（売り切れ），評估時覆蓋 cards.yaml 的行情

同一個搜尋關鍵字查兩次：
- 「關鍵字 PSA」的成交 → 依標題裡的 PSA 等級分組取中位數
- 「關鍵字」的成交 → 排除整批賣、已鑑定後當裸卡行情
標題都要再對到該卡的 match，避免把同名別版的價格混進來。
"""
from __future__ import annotations

import re
import statistics
import unicodedata
from urllib.parse import urlencode

BASE = "https://jp.mercari.com"
GRADE_RE = re.compile(r"PSA\s*(10|[6-9])(?!\d)", re.IGNORECASE)
MIN_GRADED = 2   # 某等級至少幾筆成交才採用
MIN_RAW = 3      # 裸卡至少幾筆成交才採用
MAX_SAMPLES = 10  # 每組只看最近幾筆


def sold_url(keyword: str) -> str:
    params = {"keyword": keyword, "status": "sold_out", "sort": "created_time", "order": "desc"}
    return f"{BASE}/search?{urlencode(params)}"


def _norm(title: str) -> str:
    return unicodedata.normalize("NFKC", title or "")


def summarize(graded_hits: list[dict], raw_hits: list[dict], match: str) -> dict:
    """把搜尋結果整理成 {"prices": {等級: 日圓}, "raw": 日圓 或 None, "notes": [...]}"""
    from ev_check import NOT_SINGLE_RAW

    pat = re.compile(match, re.IGNORECASE)
    groups: dict[int, list[int]] = {}
    for r in graded_hits:
        t = _norm(r["title"])
        m = GRADE_RE.search(t)
        price = int(re.sub(r"\D", "", r.get("price") or "") or 0)
        if not m or not price or not pat.search(t) or re.search(r"まとめ|セット|\d+\s*枚", t):
            continue
        g = int(m.group(1))
        if len(groups.setdefault(g, [])) < MAX_SAMPLES:
            groups[g].append(price)
    raws = []
    for r in raw_hits:
        t = _norm(r["title"])
        price = int(re.sub(r"\D", "", r.get("price") or "") or 0)
        if price and pat.search(t) and not NOT_SINGLE_RAW.search(t) and len(raws) < MAX_SAMPLES:
            raws.append(price)

    prices = {g: int(statistics.median(v)) for g, v in groups.items() if len(v) >= MIN_GRADED}
    raw = int(statistics.median(raws)) if len(raws) >= MIN_RAW else None
    parts = [f"PSA{g} ¥{p:,}（{len(groups[g])}筆）" for g, p in sorted(prices.items(), reverse=True)]
    if raw:
        parts.append(f"裸卡 ¥{raw:,}（{len(raws)}筆）")
    notes = ["Mercari 近期成交中位數：" + "、".join(parts)] if parts else []
    return {"prices": prices, "raw": raw, "notes": notes}


def live_prices(page, keyword: str, match: str) -> dict:
    from monitor import scrape

    graded = scrape(page, sold_url(f"{keyword} PSA"))
    raw = scrape(page, sold_url(keyword))
    return summarize(graded, raw, match)
