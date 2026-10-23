"""下載 Mercari 商品照片到 queue/，給標註頁匯入用（Actions 上跑，本機連不到 mercdn）"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playwright.sync_api import sync_playwright

from listing import UA, download_photos, fetch_listing, item_id


def main(text: str, out: Path) -> int:
    ids = list(dict.fromkeys(re.findall(r"m\d{8,}", text)))
    if not ids:
        print("沒有找到商品編號")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent=UA, locale="ja-JP")
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
