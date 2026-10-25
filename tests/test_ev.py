import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ev_calc import VERDICT_BUY, VERDICT_PASS, Costs, evaluate, normalize, parse_money  # noqa: E402
from ev_check import evaluate_listing, format_message, load_ev_config  # noqa: E402
from grade_estimator import Estimate, cap_at, condition_id, estimate, shift_down  # noqa: E402
from image_analysis import HeuristicAnalyzer, Observation  # noqa: E402
from listing import item_id, parse_condition, parse_price, photo_urls  # noqa: E402

FX = {"JPY": 1, "USD": 150}
SETS = {
    "optimistic": {10: 0.25, 9: 0.40, 8: 0.20, 7: 0.10, 6: 0.05},
    "base": {10: 0.10, 9: 0.35, 8: 0.30, 7: 0.15, 6: 0.07, "low": 0.03},
    "pessimistic": {10: 0.02, 9: 0.15, 8: 0.30, 7: 0.25, 6: 0.15, "low": 0.13},
}


def test_parse_money():
    assert parse_money(1000, FX) == 1000
    assert parse_money("2,861 USD", FX) == 2861 * 150
    assert parse_money("¥5,000", FX) == 5000
    with pytest.raises(ValueError):
        parse_money("10 EUR", FX)


def test_ev_matches_hand_calculation():
    costs = Costs(grading_tiers=[(None, 3000)], grading_shipping=1000, sell_fee_pct=0.1)
    probs = {"optimistic": {10: 1}, "base": {10: 0.5, 9: 0.5}, "pessimistic": {9: 1}}
    ev = evaluate(10000, probs, {10: 40000, 9: 20000}, raw_price=15000, costs=costs, min_roi=0.25)
    base = ev.scenarios["base"]
    # 收入 0.5*36000 + 0.5*18000 = 27000；成本 10000+3000+1000 = 14000
    assert base.ev_revenue == pytest.approx(27000)
    assert base.total_cost == pytest.approx(14000)
    assert base.ev_net == pytest.approx(13000)
    # raw 淨利 = 13500 - 10000 = 3500 -> 增量 9500
    assert ev.uplift == pytest.approx(9500)
    assert ev.verdict == VERDICT_BUY
    # 最高出價：悲觀 18000-(p+4000)>=0 -> p<=14000；基準 (27000-(p+4000))/(p+4000)>=0.25 -> p<=17600
    assert ev.max_bid == pytest.approx(14000, rel=1e-3)
    # 打平 PSA10：p*36000+(1-p)*18000 = 14000 -> 不需要 PSA10 就打平
    assert ev.breakeven_p10 == 0.0


def test_overpriced_is_rejected():
    costs = Costs(grading_tiers=[(None, 3000)])
    probs = {"optimistic": {10: 1}, "base": {10: 0.5, 9: 0.5}, "pessimistic": {9: 1}}
    ev = evaluate(30000, probs, {10: 40000, 9: 20000}, raw_price=15000, costs=costs)
    assert ev.verdict == VERDICT_PASS
    assert ev.max_bid is not None and ev.max_bid < 30000


def test_grading_fee_tiers():
    costs = Costs.from_config({"grading": {"tiers": [{"max_value": "499 USD", "fee": "25 USD"}, {"fee": "75 USD"}]}}, FX)
    assert costs.grading_fee(400 * 150) == 25 * 150
    assert costs.grading_fee(600 * 150) == 75 * 150


def test_shift_and_cap():
    p = normalize({10: 0.5, 9: 0.5})
    assert shift_down(p, 1)[9] == pytest.approx(0.5) and shift_down(p, 1)[8] == pytest.approx(0.5)
    assert cap_at(p, 8)[8] == pytest.approx(1.0)


def test_condition_mapping():
    assert condition_id("目立った傷や汚れなし") == 3
    assert condition_id("新品、未使用") == 1
    assert estimate(SETS, 6).scenarios is None
    good = estimate(SETS, 3).scenarios
    bad = estimate(SETS, 5).scenarios
    assert good["base"][10] > bad["base"][10]
    assert bad["base"] == pytest.approx(normalize(SETS["pessimistic"]))


def test_description_negation():
    clean = estimate(SETS, 3, description="目立った傷や汚れなし。折れはありません。スリーブ保管")
    assert not any("描述" in n for n in clean.notes)
    dirty = estimate(SETS, 3, description="角に白欠けがあります。初期傷あり")
    assert any("白邊" in n for n in dirty.notes)
    assert dirty.scenarios["base"][10] == 0


def test_observation_centering_cap():
    obs = Observation(source="t", confidence=0.9, centering_lr=63, centering_tb=52, has_back=True)
    est = estimate(SETS, 3, observations=[obs], photo_count=4)
    assert est.scenarios["optimistic"][10] == 0 and est.scenarios["optimistic"][9] == 0
    low = Observation(source="t", confidence=0.2, centering_lr=75)
    est2 = estimate(SETS, 3, observations=[low], photo_count=4)
    assert est2.scenarios["base"][10] == pytest.approx(normalize(SETS["base"])[10])


def test_few_photos_penalty():
    one = estimate(SETS, 3, photo_count=1).scenarios["base"]
    many = estimate(SETS, 3, photo_count=5).scenarios["base"]
    assert one[10] < many[10]


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def make_front(left: int, right: int, top: int = 30, bottom: int = 30) -> bytes:
    w, h = 315, 440  # 63:88
    card = np.zeros((h, w, 3))
    card[:] = (240, 200, 30)  # 黃框
    card[top:h - bottom, left:w - right] = (60, 120, 180)  # 圖面
    img = np.full((h + 120, w + 120, 3), 128.0)  # 灰色背景
    img[60:60 + h, 60:60 + w] = card
    return _png(img)


def make_back(whitening: bool) -> bytes:
    w, h = 315, 440
    card = np.zeros((h, w, 3))
    card[:] = (30, 70, 170)
    if whitening:
        for y, x in [(0, 0), (0, w - 25), (h - 25, 0), (h - 25, w - 25)]:
            card[y:y + 25, x:x + 25] = (250, 250, 250)
    img = np.full((h + 120, w + 120, 3), 128.0)
    img[60:60 + h, 60:60 + w] = card
    return _png(img)


def test_heuristic_centering_and_whitening():
    obs = HeuristicAnalyzer().analyze([make_front(20, 40), make_back(True)], {})
    assert obs.centering_lr == pytest.approx(66.7, abs=2)
    assert obs.centering_tb == pytest.approx(50, abs=2)
    assert obs.has_front and obs.has_back
    assert obs.corners in ("light", "heavy")
    clean = HeuristicAnalyzer().analyze([make_front(30, 30), make_back(False)], {})
    assert clean.centering_lr == pytest.approx(50, abs=2)
    assert clean.corners == "none"
    assert clean.confidence >= 0.5


def test_listing_parsers():
    assert item_id("https://jp.mercari.com/item/m12345678901?x=1") == "m12345678901"
    assert parse_price("¥80,000") == 80000
    body = "商品の詳細\nカテゴリー\nポケモンカード\n商品の状態\nやや傷や汚れあり\n配送料の負担"
    assert parse_condition(body) == "やや傷や汚れあり"
    html = ('<img src="https://static.mercdn.net/item/detail/orig/photos/m111_2.jpg?x">'
            '<img src="https://static.mercdn.net/item/detail/orig/photos/m111_1.jpg">'
            '<img src="https://static.mercdn.net/item/detail/orig/photos/m999_1.jpg">')
    assert photo_urls(html, "m111") == [
        "https://static.mercdn.net/item/detail/orig/photos/m111_1.jpg",
        "https://static.mercdn.net/item/detail/orig/photos/m111_2.jpg",
    ]


def test_end_to_end_with_repo_config():
    cfg = load_ev_config()
    listing = {"id": "m1", "url": "https://jp.mercari.com/item/m1", "title": "ルギア 047/P", "price": 60000,
               "condition": "目立った傷や汚れなし", "description": "スリーブ保管です"}
    photos = [make_front(20, 40), make_back(True)]
    est, ev = evaluate_listing(cfg, "lugia-e-p-047", listing, photos, analyzers=[HeuristicAnalyzer()])
    assert ev is not None
    msg = format_message("ルギア e-P 047", listing, est, ev, detail=True)
    assert "最高出價" in msg and "置中約 67/33 → 上限 PSA 7" in msg
    skip_est, skip_ev = evaluate_listing(cfg, "lugia-e-p-047", dict(listing, condition="全体的に状態が悪い"), [], analyzers=[])
    assert skip_ev is None
    assert "跳過" in format_message("x", listing, skip_est, skip_ev)


def test_claude_analyzer_parses_structured_output(monkeypatch):
    import json
    import types

    import anthropic

    from image_analysis import ClaudeVisionAnalyzer

    captured = {}
    payload = {"centering_lr": 62, "centering_tb": None, "corners": "light", "edges": "none", "surface": None,
               "crease": False, "has_front": True, "has_back": True, "authenticity_concern": False,
               "confidence": 0.8, "notes": "左右偏移約 62/38，右上角有輕微白點。"}

    class FakeMessages:
        def create(self, **kw):
            captured.update(kw)
            return types.SimpleNamespace(stop_reason="end_turn",
                                         content=[types.SimpleNamespace(type="text", text=json.dumps(payload))])

    class FakeClient:
        def __init__(self, *a, **k):
            self.beta = types.SimpleNamespace(messages=FakeMessages())

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    obs = ClaudeVisionAnalyzer().analyze([make_front(30, 30)], {"title": "t", "condition": "c", "description": "d"})
    assert obs.centering_lr == 62 and obs.corners == "light" and obs.confidence == 0.8
    assert captured["model"] == "claude-opus-5-5"
    assert captured["messages"][0]["content"][0]["source"]["media_type"] == "image/png"
    est = estimate(SETS, 3, observations=[obs], photo_count=3)
    assert est.scenarios["base"][10] == 0  # 62/38 -> 上限 PSA 8


def test_white_chip_is_not_corner_damage():
    est = estimate(SETS, 3, description="目立った傷はありませんが、裏面の角に少し白欠けがあります。")
    assert any("白邊" in n for n in est.notes)
    assert not any("角損" in n for n in est.notes)
    assert not any("刮痕" in n for n in est.notes)


def test_vision_grade_probs_blend_into_scenarios():
    vision = Observation(source="claude", confidence=0.8, has_back=True,
                         grade_probs={"10": 0.6, "9": 0.3, "8": 0.1, "7": 0, "6": 0, "5": 0, "low": 0})
    est = estimate(SETS, 3, observations=[vision], photo_count=4)
    assert est.vision_weight == pytest.approx(0.8)
    # 基準 PSA10 = 0.2*0.10 + 0.8*0.60
    assert est.scenarios["base"][10] == pytest.approx(0.2 * 0.10 + 0.8 * 0.60)
    weak = Observation(source="claude", confidence=0.3, grade_probs={"10": 1})
    assert estimate(SETS, 3, observations=[weak], photo_count=4).vision_weight == 0


def test_front_photo_not_counted_as_back():
    obs = HeuristicAnalyzer().analyze([make_front(30, 30)], {})
    assert [a["scan"].side for a in obs.annotations if a["kind"] == "scan"] == ["front"]
    assert obs.has_back is None


def test_render_report_png():
    from ev_report import render_report

    cfg = load_ev_config()
    listing = {"id": "m1", "price": 60000, "condition": "目立った傷や汚れなし", "title": "t"}
    photos = [make_front(20, 40), make_back(True)]
    est, ev = evaluate_listing(cfg, "lugia-e-p-047", listing, photos, analyzers=[HeuristicAnalyzer()])
    png = render_report("ルギア e-P 047", listing, est, ev, photos, jpy_rate=cfg["fx"]["JPY"])
    img = Image.open(io.BytesIO(png))
    assert img.format == "PNG" and img.width == 1400
    skip_est, skip_ev = evaluate_listing(cfg, "lugia-e-p-047", dict(listing, condition="全体的に状態が悪い"), [], analyzers=[])
    assert render_report("x", listing, skip_est, skip_ev, [])[:8] == b"\x89PNG\r\n\x1a\n"



# ---- 本機逐張瑕疵檢查（defect_scan）----
import synth  # noqa: E402
from defect_scan import scan_photo  # noqa: E402


def _rng():
    return np.random.default_rng(7)


def test_scan_clean_back_and_front_have_no_findings():
    for img in (synth.photo(synth.back_card(_rng())), synth.photo(synth.back_card(_rng()), angle=4),
                synth.photo(synth.front_card(_rng()))):
        s = scan_photo(synth.jpeg(img))
        assert s.edge_severity == "none" and s.corner_severity == "none" and not s.lines


def test_scan_classifies_front_with_blue_art():
    s = scan_photo(synth.jpeg(synth.photo(synth.front_card(_rng()))))
    assert s.side == "front"
    assert scan_photo(synth.jpeg(synth.photo(synth.back_card(_rng())))).side == "back"


def test_scan_edge_chips_corner_and_crease():
    rng = _rng()
    chipped = synth.back_card(rng)
    synth.chip_edges(chipped, rng)
    assert scan_photo(synth.jpeg(synth.photo(chipped))).edge_severity == "heavy"
    corner = synth.back_card(rng)
    synth.white_corner(corner)
    assert scan_photo(synth.jpeg(synth.photo(corner))).corner_severity == "heavy"
    creased = synth.back_card(rng)
    synth.crease(creased)
    s = scan_photo(synth.jpeg(synth.photo(creased)))
    assert s.crease_suspect  # 被寶可夢球切成兩段，仍合併成一條摺痕
    scratched = synth.back_card(rng)
    synth.scratch(scratched)
    s = scan_photo(synth.jpeg(synth.photo(scratched)))
    assert [l["kind"] for l in s.lines] == ["scratch"]


def test_scan_closeups():
    rng = _rng()
    corner = synth.back_card(rng)
    synth.white_corner(corner)
    clean = synth.photo(synth.back_card(rng))
    for c in ("tl", "br"):
        assert scan_photo(synth.jpeg(synth.closeup(synth.photo(corner), c))).corner_severity == "heavy"
        s = scan_photo(synth.jpeg(synth.closeup(clean, c)))
        assert s.edge_severity == "none" and s.corner_severity == "none"
        # 正面特寫：L 形黃框旁的白色圖面不能被當成白邊
        f = scan_photo(synth.jpeg(synth.closeup(synth.photo(synth.front_card(rng)), c)))
        assert f.side == "front" and f.edge_severity == "none"


def test_white_background_disables_corner_tip_check():
    s = scan_photo(synth.jpeg(synth.photo(synth.back_card(_rng()), bg=(235, 235, 235))))
    assert s.corner_severity == "none"
    assert any("背景偏白" in n for n in s.notes)


def test_local_analyzer_over_full_listing():
    rng = _rng()
    f, b = synth.front_card(rng, left=34, right=46), synth.back_card(rng)
    synth.crease(b)
    fp, bp = synth.photo(f, seed=1), synth.photo(b, seed=2)
    photos = [synth.jpeg(fp), synth.jpeg(bp)] + [synth.jpeg(synth.closeup(bp, c)) for c in ("tl", "tr", "bl", "br")]
    obs = HeuristicAnalyzer().analyze(photos, {})
    assert obs.has_front and obs.has_back
    assert obs.centering_lr == pytest.approx(57.5, abs=2)  # 特寫不拿來量置中
    assert obs.crease is True  # 全圖和特寫都看到同一條線
    assert obs.confidence >= 0.5
    est = estimate(SETS, 3, observations=[obs], photo_count=len(photos))
    assert est.scenarios["optimistic"][7] == 0 and est.scenarios["optimistic"][6] == 0  # 摺痕 -> 上限 PSA 5

    from ev_report import annotated_photos
    album = annotated_photos(photos, Estimate(est.scenarios, 3, observations=[obs]))
    assert 1 <= len(album) <= len(photos)


def test_pick_card_by_title():
    from ev_check import pick_card

    cfg = {"cards": {"a": {"match": r"098\s*/?\s*XY-?P"}, "b": {"match": r"207/XY-P"}}}
    assert pick_card(cfg, "lugia", "whatever") == "lugia"
    assert pick_card(cfg, ["a", "b"], "メガトウキョーのピカチュウ ０９８/XY-P") == "a"
    assert pick_card(cfg, ["a", "b"], "ポンチョ 207/XY-P") == "b"
    assert pick_card(cfg, ["a", "b"], "ピカチュウ XY-P 150") is None
    assert pick_card(cfg, None, "x") is None


def test_pick_card_skips_bundles_and_slabs():
    from ev_check import pick_card

    cfg = {"cards": {"a": {"match": "リザードン"}}}
    assert pick_card(cfg, "auto", "旧裏 リザードン") == "a"
    assert pick_card(cfg, "auto", "まとめ売り 旧裏 リザードン カメックス") is None
    assert pick_card(cfg, "auto", "PSA10 リザードン 旧裏") is None


def test_market_summarize_filters_by_card_and_grade():
    from market import summarize

    graded = [
        {"title": "リザード 011/T PSA10", "price": "¥50,000"},
        {"title": "リザード 011/T PSA 10 美品", "price": "¥60,000"},
        {"title": "リザード 011/T PSA9", "price": "¥20,000"},
        {"title": "リザードン 014/P PSA10", "price": "¥900,000"},
    ]
    raw = [{"title": f"リザード 011/T {i}", "price": f"¥{p}"} for i, p in enumerate((5000, 7000, 6000))]
    raw.append({"title": "リザード 011/T まとめ", "price": "¥99,999"})
    out = summarize(graded, raw, r"リザード(?!ン).*011/T")
    assert out["prices"] == {10: 55000}
    assert out["raw"] == 6000
    assert "PSA10 ¥55,000（2筆）" in out["notes"][0]


def test_live_prices_override_profile():
    from ev_check import evaluate_listing, load_ev_config

    cfg = load_ev_config()
    listing = {"title": "メガトウキョーのピカチュウ 098/XY-P", "price": 70000, "condition": 3}
    _, base = evaluate_listing(cfg, "pikachu-xy-p-098", dict(listing))
    est, live = evaluate_listing(cfg, "pikachu-xy-p-098", dict(listing),
                                 live={"prices": {10: 1}, "raw": None, "notes": ["Mercari 近期成交中位數：PSA10 ¥1（2筆）"]})
    assert live.scenarios["base"].ev_net < base.scenarios["base"].ev_net
    assert est.notes[0].startswith("Mercari 近期成交")


def test_new_back_is_back_and_logo_highlights_are_not_specks():
    s = scan_photo(synth.jpeg(synth.photo(synth.new_back_card(_rng()))))
    assert s.side == "back"
    assert s.specks == 0 and not s.lines
    assert s.edge_severity == "none"


def test_new_back_white_bottom_edge_detected():
    card = synth.new_back_card(_rng())
    synth.white_bottom_edge(card)
    s = scan_photo(synth.jpeg(synth.photo(card)))
    assert s.edge_severity == "heavy"


def test_cased_card_edges_are_not_flagged():
    s = scan_photo(synth.jpeg(synth.photo(synth.new_back_card(_rng()), bg=(120, 120, 120))))
    assert s.edge_severity == "none" and s.corner_severity == "none"
    assert any("卡套" in n for n in s.notes)


def test_old_promo_profiles_match_by_number():
    from ev_check import load_ev_config, pick_card

    cfg = load_ev_config()
    assert pick_card(cfg, "auto", "ピカチュウM LV.X 043/DPt-P") == "pikachu-m-lvx-dpt-p-043"
    assert pick_card(cfg, "auto", "ポケパークのピカチュウ 43/PCG-P") == "pokeparks-pikachu-pcg-p-043"
    assert pick_card(cfg, "auto", "旧裏 ミュウ CDプロモ 光沢") == "mew-cd-promo"
    assert pick_card(cfg, "auto", "ピカチュウ 004/P ANA") == "pikachu-e-p-004"  # 舊的 e 卡 profile 仍優先
    assert pick_card(cfg, "auto", "アチャモ 052/ADV-P") is None
    assert pick_card(cfg, "auto", "マリオピカチュウ 294/XY-P") == "mario-pikachu-xy-p-294"
    assert pick_card(cfg, "auto", "ピカチュウ 002/SM-P") is None  # 002/S-P 是シブヤ，不能混到 SM-P
