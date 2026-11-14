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


# Mercari 對海外連線（例如 GitHub Actions 在美國）會把價格顯示成當地幣別，如「US$ 131.39」。
# 換回日圓用 Mercari 自己的匯率（頁面會呼叫 getCurrencyConversionRate，watch_conversion_rate 會記下來）；
# 還沒抓到時用 FALLBACK_RATES（1 日圓 = 多少外幣，由 monitor 依 cards.yaml 的 fx 設定）。
CONVERSION_RATES: dict[str, float] = {}
FALLBACK_RATES: dict[str, float] = {}
FOREIGN_PRICE_RE = re.compile(r"(US|HK|NT)\s*\$\s*([\d,]+(?:\.\d+)?)")
FOREIGN_CODES = {"US": "USD", "HK": "HKD", "NT": "TWD"}


def watch_conversion_rate(ctx) -> None:
    """在 Playwright context 上記錄 Mercari 的外幣匯率"""

    def on_response(resp):
        if "getCurrencyConversionRate" not in resp.url:
            return
        try:
            data = resp.json()
            if data.get("currencyCode") and float(data.get("rate") or 0) > 0:
                CONVERSION_RATES[data["currencyCode"]] = float(data["rate"])
        except Exception:
            pass

    ctx.on("response", on_response)


def set_fallback_rates(fx: dict) -> None:
    """cards.yaml 的 fx（1 外幣 = 多少台幣）→ 1 日圓 = 多少外幣"""
    jpy = float(fx.get("JPY") or 0)
    if not jpy:
        return
    for code in ("USD", "HKD", "TWD"):
        if fx.get(code):
            FALLBACK_RATES[code] = jpy / float(fx[code])


def jp_context(browser):
    """日文、東京時區的 context，並記錄 Mercari 外幣匯率"""
    ctx = browser.new_context(locale="ja-JP", timezone_id="Asia/Tokyo", user_agent=UA)
    watch_conversion_rate(ctx)
    return ctx


def parse_price(text: str) -> int | None:
    """回傳日圓；頁面若只有外幣價就換回日圓"""
    m = re.search(r"[¥￥]\s*([\d,]+)|([\d,]+)\s*円", text or "")
    if m:
        return int((m.group(1) or m.group(2)).replace(",", ""))
    m = FOREIGN_PRICE_RE.search(text or "")
    if not m:
        return None
    code = FOREIGN_CODES[m.group(1)]
    rate = CONVERSION_RATES.get(code) or FALLBACK_RATES.get(code)
    if not rate:
        return None
    return round(float(m.group(2).replace(",", "")) / rate)


def strip_price(text: str) -> str:
    """把搜尋結果文字裡的價格拿掉，只留標題"""
    text = FOREIGN_PRICE_RE.sub("", text or "")
    text = re.sub(r"[¥￥]\s*[\d,]+|[\d,]+\s*円", "", text)
    return re.sub(r"\s+", " ", text).strip()


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
