
"""Mercari 新上架監控 -> Telegram 通知

流程：每個品項用「最新上架」排序搜尋 -> 與 seen.json 比對 -> 新商品推送。
第一次看到某品項時只記錄、不通知，避免一次洗版。
"""
import json
import os
import random
import re
import time
from pathlib import Path
from urllib.parse import urlencode

import requests
import yaml
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent
BASE = "https://jp.mercari.com"
SEEN_PATH = ROOT / "seen.json"
MAX_SEEN_PER_ITEM = 500
MAX_NOTIFY_PER_ITEM = 10

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")


def notify(text: str) -> None:
    if not (TG_TOKEN and TG_CHAT):
        print("[DRY-RUN]", text)
        return
    r = requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        json={"chat_id": TG_CHAT, "text": text, "disable_web_page_preview": False},
        timeout=20,
    )
    if not r.ok:
        print("Telegram 發送失敗:", r.status_code, r.text)


def build_url(item: dict, defaults: dict) -> str:
    conditions = item.get("conditions", defaults["conditions"])
    params = {
        "keyword": item["keyword"],
        "status": "on_sale",
        "sort": "created_time",
        "order": "desc",
        "item_condition_id": ",".join(map(str, conditions)),
    }
    max_price = item.get("max_price", defaults.get("max_price"))
    if max_price:
        params["price_max"] = max_price
    return f"{BASE}/search?{urlencode(params)}"


def scrape(page, url: str) -> list[dict]:
    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    try:
        page.wait_for_selector('a[href*="/item/m"]', timeout=20000)
    except Exception:
        return []  # 沒結果，或被擋

    found: dict[str, dict] = {}
    for a in page.query_selector_all('a[href*="/item/m"]'):
        href = a.get_attribute("href") or ""
        m = re.search(r"/item/(m\d+)", href)
        if not m or m.group(1) in found:
            continue
        text = (a.get_attribute("aria-label") or a.inner_text() or "")
        text = re.sub(r"\s+", " ", text).strip()
        price = re.search(r"[¥￥]\s*([\d,]+)|([\d,]+)\s*円", text)
        price_txt = ""
        if price:
            price_txt = "¥" + (price.group(1) or price.group(2))
        found[m.group(1)] = {
            "id": m.group(1),
            "title": text,
            "price": price_txt,
            "url": f"{BASE}/item/{m.group(1)}",
        }
    return list(found.values())[:40]


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    defaults = cfg["defaults"]
    exclude = [w.lower() for w in defaults.get("exclude_words", [])]
    seen: dict[str, list[str]] = (
        json.loads(SEEN_PATH.read_text(encoding="utf-8")) if SEEN_PATH.exists() else {}
    )

    total_results = 0
    new_count = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            locale="ja-JP",
            timezone_id="Asia/Tokyo",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = ctx.new_page()

        for item in cfg["items"]:
            key = item["name"]
            try:
                results = scrape(page, build_url(item, defaults))
            except Exception as e:  # 單一品項失敗不影響其他
                print(f"[{key}] 抓取失敗: {e}")
                continue
            total_results += len(results)
            print(f"[{key}] 抓到 {len(results)} 件")

            first_time = key not in seen
            known = set(seen.get(key, []))
            fresh = [r for r in results if r["id"] not in known]

            if not first_time:
                shown = 0
                for r in fresh:
                    if any(w in r["title"].lower() for w in exclude):
                        continue
                    if shown >= MAX_NOTIFY_PER_ITEM:
                        break
                    notify(f"🆕 {key}\n{r['title']}\n{r['price']}\n{r['url']}")
                    shown += 1
                    new_count += 1

            merged = [r["id"] for r in results] + [i for i in seen.get(key, []) if i not in {r["id"] for r in results}]
            seen[key] = merged[:MAX_SEEN_PER_ITEM]
            time.sleep(random.uniform(2, 4))

        browser.close()

    # 全部品項都 0 筆，八成是被 Mercari 擋掉或頁面改版
    if total_results == 0:
        notify("⚠️ Mercari 監控：這次所有品項都抓到 0 件，可能被擋或網頁改版，請檢查。")

    SEEN_PATH.write_text(json.dumps(seen, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"完成：新通知 {new_count} 件")


if __name__ == "__main__":
    main()
