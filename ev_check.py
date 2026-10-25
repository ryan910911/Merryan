"""單張刊登的鑑定期望值評估

用法：
  python ev_check.py https://jp.mercari.com/item/m12345678901 --card lugia-e-p-047
  python ev_check.py --card lugia-e-p-047 --price 80000 --condition 3 --photos front.jpg back.jpg
  python ev_check.py --card lugia-e-p-047 --listing listing.json --log ev_log.csv
  python ev_check.py <網址> --card lugia-e-p-047 --report report.png   # 輸出視覺化報告

monitor.py 也會呼叫 evaluate_listing() / format_message()，讓有設定 card 的品項
在 Telegram 通知裡直接附上 EV 判斷。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ev_calc import GRADES, VERDICT_BUY, VERDICT_RISKY, VERDICT_SKIP, Costs, Evaluation, evaluate, parse_money
from grade_estimator import CONDITION_LABELS, Estimate, estimate
from image_analysis import build_analyzers

ROOT = Path(__file__).parent
CARDS_PATH = ROOT / "cards.yaml"
SCEN_ZH = {"optimistic": "樂觀", "base": "基準", "pessimistic": "悲觀"}
VERDICT_ICON = {VERDICT_BUY: "💰", VERDICT_RISKY: "🤔", VERDICT_SKIP: "⏭️"}


def load_ev_config(path: Path = CARDS_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def card_profile(cfg: dict, key: str) -> dict:
    cards = cfg.get("cards", {}) or {}
    if key not in cards:
        raise KeyError(f"cards.yaml 找不到卡種 {key!r}（可用：{', '.join(cards)}）")
    return cards[key]


NOT_SINGLE_RAW = re.compile(r"まとめ|セット|\d+\s*枚|オリパ|PSA\s*\d|BGS|CGC|ARS|鑑定済|鑑定品", re.IGNORECASE)


def pick_card(cfg: dict, spec, title: str) -> str | None:
    """依品項的 card 設定挑卡種：字串直接用；auto 或清單則用各卡 match（正規式）比對標題，第一個符合的勝出"""
    if not spec:
        return None
    if spec == "auto":  # 用 cards.yaml 全部卡的 match 比對
        spec = list(cfg.get("cards") or {})
    elif isinstance(spec, str):
        return spec
    norm = unicodedata.normalize("NFKC", title or "")
    if NOT_SINGLE_RAW.search(norm):  # 多張一起賣或已鑑定的卡，不適用裸卡鑑定評估
        return None
    for key in spec:
        pattern = card_profile(cfg, key).get("match")
        if pattern and re.search(pattern, norm, re.IGNORECASE):
            return key
    return None


def evaluate_listing(cfg: dict, card_key: str, listing: dict, photos: list[bytes] | None = None, analyzers=None,
                     live: dict | None = None):
    """回傳 (Estimate, Evaluation 或 None)"""
    fx = cfg["fx"]
    card = card_profile(cfg, card_key)
    sets = card.get("probabilities") or cfg["probability_sets"][card.get("probability_set", "default")]
    photos = photos or []
    if analyzers is None:
        analyzers = build_analyzers(cfg.get("analysis"))
    observations = []
    if photos:
        for a in analyzers:
            try:
                observations.append(a.analyze(photos, listing))
            except Exception as e:  # 單一 analyzer 壞掉不影響評估
                print(f"[{a.name}] 照片分析失敗: {e}")
    photo_count = len(photos) if photos else len(listing.get("photos") or []) or None
    est = estimate(
        sets,
        listing.get("condition"),
        description=listing.get("description", ""),
        title=listing.get("title", ""),
        observations=observations,
        photo_count=photo_count,
        min_confidence=float((cfg.get("analysis") or {}).get("min_confidence", 0.5)),
    )
    if est.scenarios is None or not listing.get("price"):
        return est, None

    # 「不鑑定、直接賣裸卡」的比較基準：新品〜未使用に近い 用 A 品相價，其餘用有使用痕跡的裸卡價
    raw_key = "raw_price" if est.condition in (1, 2) or not card.get("raw_price_played") else "raw_price_played"
    if not card.get(raw_key):
        raw_key = "raw_price_played"
    raw_price = parse_money(card[raw_key], fx)
    prices = {g: parse_money(v, fx) for g, v in (card.get("prices") or {}).items() if v not in (None, "")}
    if live:  # Mercari 近期成交（日圓）優先於 cards.yaml 的行情
        for g, jpy in (live.get("prices") or {}).items():
            prices[g] = parse_money(f"{jpy} JPY", fx)
        if live.get("raw"):
            live_raw = parse_money(f"{live['raw']} JPY", fx)
            top = max((prices[g] for g in (10, 9) if g in prices), default=None)
            if top is not None and live_raw >= top:  # 裸卡比 PSA 9/10 還貴，八成混到別的商品
                est.notes.append("近期裸卡成交高於 PSA 9/10 行情，判定樣本有誤，沿用 cards.yaml 裸卡價")
            else:
                raw_price = live_raw
        est.notes[:0] = live.get("notes") or []
    low = parse_money(card["low_grade_price"], fx) if card.get("low_grade_price") else None
    ev = evaluate(
        price=parse_money(f"{listing['price']} JPY", fx),  # Mercari 價格是日圓
        scenario_probs=est.scenarios,
        prices=prices,
        raw_price=raw_price,
        costs=Costs.from_config(cfg.get("costs", {}), fx),
        min_roi=float((cfg.get("decision") or {}).get("min_roi", 0.25)),
        low_price=low,
    )
    return est, ev


def money(v: float | None) -> str:
    """計價幣別（cards.yaml 的 fx 裡匯率為 1 的那個，目前是台幣）"""
    if v is None:
        return "—"
    return f"{'-' if v < 0 else ''}NT${abs(v):,.0f}"


def yen(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{'-' if v < 0 else ''}¥{abs(v):,.0f}"


def format_message(card_name: str, listing: dict, est: Estimate, ev: Evaluation | None, detail: bool = False,
                   jpy_rate: float | None = None) -> str:
    """jpy_rate：1 日圓 = 多少計價幣別，用來把最高出價換回日圓"""
    cond = CONDITION_LABELS.get(est.condition, listing.get("condition") or "狀態不明")
    price_jpy = listing.get("price")
    price_txt = yen(price_jpy) + (f"（{money(price_jpy * jpy_rate)}）" if price_jpy and jpy_rate else "")
    head = f"{listing.get('title') or card_name}\n{price_txt} · {cond}"
    if ev is None:
        reason = est.notes[0] if est.scenarios is None else "抓不到價格"
        return f"⏭️ {VERDICT_SKIP} | {card_name}\n{head}\n{reason}\n{listing.get('url', '')}".strip()

    s = ev.scenarios
    lines = [
        f"{VERDICT_ICON.get(ev.verdict, '🙅')} {ev.verdict} | {card_name}",
        head,
        f"期望淨利 基準 {money(s['base'].ev_net)}（ROI {s['base'].roi:.0%}）/ 悲觀 {money(s['pessimistic'].ev_net)} / 樂觀 {money(s['optimistic'].ev_net)}",
        f"鑑定增量 {money(ev.uplift)} · PSA10 機率 {s['base'].probs[10]:.0%}"
        + (f"（打平需 {ev.breakeven_p10:.0%}）" if ev.breakeven_p10 is not None else "（PSA10 也無法打平）"),
        "最高出價 " + ((yen(ev.max_bid / jpy_rate) if jpy_rate else money(ev.max_bid)) if ev.max_bid is not None
                      else "—（鑑定不如直接賣裸卡）" if ev.uplift <= 0 else "—（任何價格都達不到門檻）"),
    ]
    if ev.reasons:
        lines.append("原因：" + "；".join(ev.reasons))
    shown = est.notes if detail else est.notes[:6]
    if shown:
        lines.append("依據：\n- " + "\n- ".join(shown))
    if detail:
        lines.append(f"\n成本：到手 {money(ev.acquisition)} + 鑑定費 {money(ev.grading_fee)} + 往返運保 {money(s['base'].total_cost - ev.acquisition - ev.grading_fee)}")
        lines.append(f"裸卡直接賣淨利 {money(ev.raw_net)}")
        header = "等級    " + "  ".join(f"{SCEN_ZH[k]:>4}" for k in ("optimistic", "base", "pessimistic"))
        lines.append(header)
        for g in GRADES:
            label = f"PSA {g}" if g != "low" else "PSA ≤4"
            lines.append(f"{label:<7} " + "  ".join(f"{s[k].probs[g]:>5.0%}" for k in ("optimistic", "base", "pessimistic")))
    lines.append(listing.get("url", ""))
    return "\n".join(lines).strip()


def append_log(path: Path, card_key: str, listing: dict, est: Estimate, ev: Evaluation | None) -> None:
    """記錄預估，鑑定回來後自己補 actual_grade 欄位，用來校準機率"""
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "card", "item", "price", "condition", "verdict", "base_ev_net", "base_roi", "pess_ev_net", "uplift", "max_bid"]
                       + [f"p_{g}" for g in GRADES] + ["actual_grade"])
        base = ev.scenarios["base"] if ev else None
        w.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"), card_key, listing.get("id") or listing.get("url"),
            listing.get("price"), est.condition, ev.verdict if ev else VERDICT_SKIP,
            round(base.ev_net) if base else "", round(base.roi, 3) if base else "",
            round(ev.scenarios["pessimistic"].ev_net) if ev else "", round(ev.uplift) if ev else "",
            round(ev.max_bid) if ev and ev.max_bid is not None else "",  # 計價幣別
        ] + [round(base.probs[g], 3) if base else "" for g in GRADES] + [""])


def _items() -> list[dict]:
    import yaml

    return yaml.safe_load(Path(__file__).with_name("config.yaml").read_text(encoding="utf-8")).get("items", [])


def watched_by(title: str) -> list[str]:
    """粗估標題會被 config.yaml 哪些品項搜到：關鍵字每個詞都出現在標題裡"""
    items = _items()
    norm = unicodedata.normalize("NFKC", title).lower()
    return [
        it["name"]
        for it in items
        if all(unicodedata.normalize("NFKC", w).lower() in norm for w in it["keyword"].split())
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Mercari 裸卡鑑定期望值評估")
    ap.add_argument("url", nargs="?", help="Mercari 商品網址或編號（會用 Playwright 開頁面）")
    ap.add_argument("--card", help="cards.yaml 裡的卡種 key；不給就依標題卡號自動挑")
    ap.add_argument("--listing", help="改用 JSON 檔提供刊登資料（title/price/condition/description）")
    ap.add_argument("--price", type=float, help="覆寫買入價（日圓）")
    ap.add_argument("--condition", help="覆寫商品狀態（1~6 或日文字樣）")
    ap.add_argument("--photos", nargs="*", default=[], help="本機照片檔")
    ap.add_argument("--config", default=str(CARDS_PATH))
    ap.add_argument("--log", help="把結果附加到 CSV")
    ap.add_argument("--report", help="輸出視覺化報告 PNG 的路徑")
    ap.add_argument("--no-live", action="store_true", help="不查 Mercari 近期成交價")
    ap.add_argument("--telegram", action="store_true", help="把結果、報告圖與瑕疵照片傳到 Telegram")
    args = ap.parse_args(argv)

    cfg = load_ev_config(Path(args.config))
    listing: dict = {}
    photos = [Path(p).read_bytes() for p in args.photos]
    if args.listing:
        listing = json.loads(Path(args.listing).read_text(encoding="utf-8"))
    elif args.url:
        from playwright.sync_api import sync_playwright

        from listing import UA, download_photos, fetch_listing

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_context(locale="ja-JP", timezone_id="Asia/Tokyo", user_agent=UA).new_page()
            listing = fetch_listing(page, args.url)
            browser.close()
        if not photos:
            photos = download_photos(listing["photos"])
    if args.price is not None:
        listing["price"] = args.price
    if args.condition:
        listing["condition"] = int(args.condition) if args.condition.isdigit() else args.condition
    if not listing.get("price"):
        ap.error("沒有價格：請給商品網址、--listing 或 --price")

    title = listing.get("title", "")
    if title:
        print("標題：", title)
        hits = watched_by(title)
        print("會被哪些監控品項抓到（依關鍵字粗估）：", "、".join(hits) if hits else "沒有")
    card_key = args.card or pick_card(cfg, list(cfg.get("cards") or {}), title)
    if not card_key:
        msg = f"⚠️ 標題對不到 cards.yaml 裡任何卡號，無法評估：{title}"
        print(msg)
        if args.telegram:
            from monitor import notify

            notify(f"{msg}\n{args.url or ''}")
        return 0  # 對不到卡不算錯誤，避免 GitHub 寄失敗通知
    print("使用行情：", card_key)
    live = None
    if args.url and not args.no_live:
        hits = watched_by(title)
        keyword = card_profile(cfg, card_key).get("sold_keyword")
        if not keyword:
            keyword = next(it["keyword"] for it in _items() if it["name"] == hits[0]) if hits else card_profile(cfg, card_key).get("name", "").split("（")[0]
        try:
            from playwright.sync_api import sync_playwright

            from listing import UA
            from market import live_prices

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_context(locale="ja-JP", timezone_id="Asia/Tokyo", user_agent=UA).new_page()
                live = live_prices(page, keyword, card_profile(cfg, card_key)["match"])
                browser.close()
            print("近期成交：", live["notes"] or f"「{keyword}」查不到足夠成交，用 cards.yaml 行情")
        except Exception as e:
            print("查近期成交失敗：", e)
    est, ev = evaluate_listing(cfg, card_key, listing, photos, live=live)
    name = card_profile(cfg, card_key).get("name", card_key)
    msg = format_message(name, listing, est, ev, detail=True, jpy_rate=cfg["fx"]["JPY"])
    print(msg)
    png = None
    if args.report or args.telegram:
        from ev_report import render_report

        png = render_report(name, listing, est, ev, photos, jpy_rate=cfg["fx"]["JPY"])
    if args.report:
        Path(args.report).write_bytes(png)
        print("報告圖：", args.report)
    if args.telegram:
        from ev_report import annotated_photos
        from monitor import notify_album, notify_photo

        url = args.url if args.url and args.url.startswith("http") else f"https://jp.mercari.com/item/{args.url or ''}"
        notify_photo(png, f"🔎 手動評估\n{msg}\n{url}")
        notify_album(annotated_photos(photos, est), f"偵測到瑕疵的照片\n{url}")
    if args.log:
        append_log(Path(args.log), card_key, listing, est, ev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
