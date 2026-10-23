"""下載 Mercari 商品照片到 queue/，給標註頁匯入用（Actions 上跑，本機連不到 mercdn）"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playwright.sync_api import sync_playwright

from listing import UA, download_photos, fetch_listing, item_id


TW_RE = re.compile(r"https?://tw\.mercari\.com/\S*?items/([0-9a-f-]{36})")


def resolve_tw(page, url: str, out: Path) -> tuple[str | None, int]:
    """台灣版網址：先找頁面裡的日本商品編號；找不到就直接下載頁面上的大圖"""
    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(6000)
    html = page.content()
    ids = re.findall(r"m\d{10,12}", html)
    if ids:
        iid = max(set(ids), key=ids.count)
        print("台灣版", url, "→", iid)
        return iid, 0
    srcs = page.eval_on_selector_all(
        "img", "els => els.filter(e => e.naturalWidth >= 400).map(e => e.currentSrc || e.src)")
    srcs = list(dict.fromkeys(srcs))
    uid = TW_RE.search(url).group(1)[:8]
    photos = download_photos(srcs, limit=20)
    for n, data in enumerate(photos, 1):
        (out / f"tw{uid}_{n}.jpg").write_bytes(data)
    title = page.title()
    (out / f"tw{uid}.json").write_text(json.dumps({"id": "tw" + uid, "url": url, "title": title, "photos": len(photos)},
                                                  ensure_ascii=False), encoding="utf-8")
    print(f"台灣版 {url}: 找不到日本編號，直接抓 {len(photos)} 張大圖 {title}")
    if not photos:
        print(html[:3000])
    return None, len(photos)


def main(text: str, out: Path) -> int:
    ids = list(dict.fromkeys(re.findall(r"(?<![0-9a-z-])m\d{8,}", text)))
    tw = [m.group(0) for m in TW_RE.finditer(text)]
    if not ids and not tw:
        print("沒有找到商品編號")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent=UA, locale="ja-JP")
        for u in tw:
            try:
                iid, n = resolve_tw(page, u, out)
            except Exception as e:
                print("台灣版頁面讀取失敗:", u, e)
                continue
            total += n
            if iid and iid not in ids:
                ids.append(iid)
        for iid in ids:
            try:
                lst = fetch_listing(page, iid)
            except Exception as e:  # 單一商品失敗不影響其他
                print("商品頁讀取失敗:", iid, e)
                continue
            photos = download_photos(lst["photos"], limit=20)
            for n, data in enumerate(photos, 1):
                (out / f"{item_id(iid)}_{n}.jpg").write_bytes(data)
            (out / f"{iid}.json").write_text(json.dumps(
                {"id": iid, "title": lst["title"], "price": lst["price"], "photos": len(photos)},
                ensure_ascii=False), encoding="utf-8")
            print(f"{iid}: {len(photos)} 張 {lst['title']}")
            total += len(photos)
        browser.close()
    print("共", total, "張")
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:]), Path("queue")))
