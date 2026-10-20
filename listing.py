"""抓 Mercari 單一商品頁：標題、價格、商品狀態、描述、照片"""
from __future__ import annotations

import re

import requests

from grade_estimator import CONDITION_LABELS

BASE = "https://jp.mercari.com"
PHOTO_RE = re.compile(r"https://static\.mercdn\.net/item/detail/orig/photos/(m\d+)_(\d+)\.jpg")
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def item_id(url_or_id: str) -> str:
    m = re.search(r"(m\d{8,})", url_or_id)
    if not m:
        raise ValueError(f"看不出商品編號：{url_or_id}")
    return m.group(1)


def parse_price(text: str) -> int | None:
    m = re.search(r"[¥￥]\s*([\d,]+)|([\d,]+)\s*円", text or "")
    if not m:
        return None
    return int((m.group(1) or m.group(2)).replace(",", ""))


def parse_condition(text: str) -> str | None:
    """在頁面文字裡找「商品の状態」後面的狀態字樣"""
    text = text or ""
    start = text.find("商品の状態")
    scope = text[start:start + 200] if start >= 0 else text
    hits = [(scope.find(label), label) for label in CONDITION_LABELS.values() if label in scope]
    return min(hits)[1] if hits else None


def photo_urls(html: str, iid: str) -> list[str]:
    nums = sorted({int(n) for i, n in PHOTO_RE.findall(html) if i == iid})
    return [f"https://static.mercdn.net/item/detail/orig/photos/{iid}_{n}.jpg" for n in nums]


def fetch_listing(page, url_or_id: str) -> dict:
    """用現有的 Playwright page 打開商品頁並解析"""
    iid = item_id(url_or_id)
    url = f"{BASE}/item/{iid}"
    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    try:
        page.wait_for_selector('[data-testid="price"], h1', timeout=20000)
        page.wait_for_timeout(1500)  # 讓圖片與描述載完
    except Exception:
        pass

    def text_of(selector: str) -> str:
        el = page.query_selector(selector)
        return re.sub(r"[ \t]+", " ", el.inner_text()).strip() if el else ""

    body = page.inner_text("body")
    title = text_of('[data-testid="name"] h1') or text_of("h1")
    price = parse_price(text_of('[data-testid="price"]')) or parse_price(body)
    description = text_of('[data-testid="description"]') or text_of("pre")
    condition = text_of('[data-testid="商品の状態"]') or parse_condition(body)
    return {
        "id": iid,
        "url": url,
        "title": title,
        "price": price,
        "condition": parse_condition(condition) or condition,
        "description": description,
        "photos": photo_urls(page.content(), iid),
    }


def download_photos(urls: list[str], limit: int = 10) -> list[bytes]:
    out = []
    for u in urls[:limit]:
        try:
            r = requests.get(u, headers={"User-Agent": UA, "Referer": BASE}, timeout=20)
            if r.ok and r.content:
                out.append(r.content)
        except requests.RequestException as e:
            print("照片下載失敗:", u, e)
    return out
