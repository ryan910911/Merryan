"""PSA 鑑定期望值（EV）計算

所有金額內部一律換成日圓計算。config 裡的金額可以寫數字（視為日圓）或
"2861 USD" 這種「數字 + 幣別」字串，依 cards.yaml 的 fx 匯率換算。

EV 收入   = Σ(各級機率 × 該級淨賣價)
EV 淨利   = EV 收入 − 總成本（買入 + 買入端費用 + 鑑定費 + 往返運費保險）
raw 淨利  = 裸卡淨賣價 − 買入成本
鑑定增量 = EV 淨利 − raw 淨利（與買入價無關，只回答「鑑定值不值得」）
ROI       = EV 淨利 / 總成本
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# 由高到低；"low" = PSA 4 以下
GRADES: list = [10, 9, 8, 7, 6, 5, "low"]

VERDICT_BUY = "建議買進"
VERDICT_RISKY = "可考慮（悲觀會虧）"
VERDICT_PASS = "不建議"
VERDICT_SKIP = "跳過"


def parse_money(value, fx: dict) -> float:
    """1234 / "1,234" / "2861 USD" / "¥5,000" -> 日圓"""
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(",", "").replace("¥", "").replace("￥", "")
    m = re.fullmatch(r"(-?[\d.]+)\s*([A-Za-z]{3})?", s)
    if not m:
        raise ValueError(f"看不懂的金額：{value!r}")
    amount = float(m.group(1))
    cur = (m.group(2) or "JPY").upper()
    if cur not in fx:
        raise ValueError(f"cards.yaml 的 fx 沒有 {cur} 匯率")
    return amount * float(fx[cur])


def grade_key(g):
    """yaml 讀進來可能是 10 或 "10"，統一成 int 或 "low" """
    if isinstance(g, str) and g.strip().lstrip("-").isdigit():
        g = int(g)
    if isinstance(g, int) and g <= 4:
        return "low"
    return g


def normalize(probs: dict) -> dict:
    out = {g: 0.0 for g in GRADES}
    for g, p in probs.items():
        out[grade_key(g)] += float(p)
    total = sum(out.values())
    if total <= 0:
        raise ValueError("機率總和為 0")
    return {g: p / total for g, p in out.items()}


@dataclass
class Costs:
    """已換算成日圓的成本設定"""
    proxy_fee_pct: float = 0.0
    proxy_fee_fixed: float = 0.0
    domestic_shipping: float = 0.0
    intl_shipping: float = 0.0
    import_duty_pct: float = 0.0
    fx_loss_pct: float = 0.0
    grading_tiers: list = field(default_factory=list)  # [(申報價值上限, 費用)]，上限 None = 無上限
    grading_shipping: float = 0.0
    sell_fee_pct: float = 0.0
    sell_shipping: float = 0.0

    @classmethod
    def from_config(cls, cfg: dict, fx: dict) -> "Costs":
        buy = cfg.get("buy", {}) or {}
        grading = cfg.get("grading", {}) or {}
        sell = cfg.get("sell", {}) or {}
        tiers = []
        for t in grading.get("tiers", []) or []:
            cap = t.get("max_value")
            tiers.append((parse_money(cap, fx) if cap not in (None, "") else None, parse_money(t["fee"], fx)))
        if not tiers and grading.get("fee") is not None:
            tiers = [(None, parse_money(grading["fee"], fx))]
        return cls(
            proxy_fee_pct=float(buy.get("proxy_fee_pct", 0) or 0),
            proxy_fee_fixed=parse_money(buy.get("proxy_fee_fixed"), fx),
            domestic_shipping=parse_money(buy.get("domestic_shipping"), fx),
            intl_shipping=parse_money(buy.get("intl_shipping"), fx),
            import_duty_pct=float(buy.get("import_duty_pct", 0) or 0),
            fx_loss_pct=float(buy.get("fx_loss_pct", 0) or 0),
            grading_tiers=tiers,
            grading_shipping=parse_money(grading.get("shipping_insurance"), fx),
            sell_fee_pct=float(sell.get("fee_pct", 0) or 0),
            sell_shipping=parse_money(sell.get("shipping"), fx),
        )

    def acquisition(self, price: float) -> float:
        """買到手（含代購、運費、關稅、匯損）的成本"""
        goods = price * (1 + self.proxy_fee_pct) + self.proxy_fee_fixed + self.domestic_shipping
        return goods * (1 + self.fx_loss_pct) + self.intl_shipping + price * self.import_duty_pct

    def grading_fee(self, declared_value: float) -> float:
        for cap, fee in self.grading_tiers:
            if cap is None or declared_value <= cap:
                return fee
        return self.grading_tiers[-1][1] if self.grading_tiers else 0.0

    def net_sale(self, gross: float) -> float:
        return gross * (1 - self.sell_fee_pct) - self.sell_shipping


@dataclass
class ScenarioResult:
    name: str
    probs: dict
    ev_revenue: float
    total_cost: float
    ev_net: float
    roi: float
    psa10_share: float  # EV 收入中來自 PSA 10 的比例


@dataclass
class Evaluation:
    price: float
    scenarios: dict  # name -> ScenarioResult
    acquisition: float
    grading_fee: float
    raw_net: float
    uplift: float
    verdict: str
    reasons: list
    max_bid: float | None
    breakeven_p10: float | None


def _grade_price(prices: dict, g, fallback: float) -> float:
    if g in prices and prices[g] is not None:
        return prices[g]
    return fallback


def gross_prices(prices: dict, low_price: float) -> dict:
    """各級賣價；沒有行情的級數用低分價補，但不高於上一個較高級數的價格"""
    out, ceiling = {}, float("inf")
    for g in GRADES:
        v = _grade_price(prices, g, min(low_price, ceiling))
        out[g] = v
        ceiling = min(ceiling, v)
    return out


def scenario(name: str, probs: dict, prices: dict, low_price: float, price: float, costs: Costs) -> ScenarioResult:
    probs = normalize(probs)
    gross = gross_prices(prices, low_price)
    revenue_by_grade = {g: probs[g] * costs.net_sale(gross[g]) for g in GRADES}
    ev_revenue = sum(revenue_by_grade.values())
    declared = sum(probs[g] * gross[g] for g in GRADES)
    total_cost = costs.acquisition(price) + costs.grading_fee(declared) + costs.grading_shipping
    ev_net = ev_revenue - total_cost
    return ScenarioResult(
        name=name,
        probs=probs,
        ev_revenue=ev_revenue,
        total_cost=total_cost,
        ev_net=ev_net,
        roi=ev_net / total_cost if total_cost else 0.0,
        psa10_share=(revenue_by_grade[10] / ev_revenue) if ev_revenue > 0 else 0.0,
    )


def _decide(scen: dict, uplift: float, min_roi: float) -> tuple[str, list]:
    base, pess = scen["base"], scen["pessimistic"]
    reasons = []
    if base.roi < min_roi:
        reasons.append(f"基準 ROI {base.roi:.0%} 低於門檻 {min_roi:.0%}")
    if uplift <= 0:
        reasons.append("鑑定增量 ≤ 0（直接賣裸卡比較划算）")
    if reasons:
        return VERDICT_PASS, reasons
    if pess.ev_net < 0:
        return VERDICT_RISKY, ["悲觀情境會虧損"]
    return VERDICT_BUY, []


def evaluate(
    price: float,
    scenario_probs: dict,
    prices: dict,
    raw_price: float,
    costs: Costs,
    min_roi: float = 0.25,
    low_price: float | None = None,
) -> Evaluation:
    """scenario_probs: {"optimistic": {...}, "base": {...}, "pessimistic": {...}}"""
    prices = {grade_key(g): v for g, v in prices.items()}
    if low_price is None:
        low_price = raw_price * 0.8

    def run(p: float) -> dict:
        return {n: scenario(n, pr, prices, low_price, p, costs) for n, pr in scenario_probs.items()}

    scen = run(price)
    acq = costs.acquisition(price)
    raw_net = costs.net_sale(raw_price) - acq
    base = scen["base"]
    uplift = base.ev_net - raw_net
    verdict, reasons = _decide(scen, uplift, min_roi)
    if base.psa10_share > 0.6:
        reasons.append(f"EV 有 {base.psa10_share:.0%} 靠 PSA 10，風險高")

    max_bid = _max_bid(run, uplift, min_roi, upper=max(price, raw_price, max(prices.values(), default=0)) * 3)
    breakeven = _breakeven_p10(scenario_probs["base"], prices, low_price, price, costs)
    return Evaluation(
        price=price,
        scenarios=scen,
        acquisition=acq,
        grading_fee=base.total_cost - acq - costs.grading_shipping,
        raw_net=raw_net,
        uplift=uplift,
        verdict=verdict,
        reasons=reasons,
        max_bid=max_bid,
        breakeven_p10=breakeven,
    )


def _max_bid(run, uplift: float, min_roi: float, upper: float) -> float | None:
    """同時滿足「基準 ROI ≥ 門檻、悲觀不虧」的最高買入價（成本對價格單調遞增，二分搜尋）"""
    if uplift <= 0:
        return None

    def ok(p: float) -> bool:
        s = run(p)
        return s["base"].roi >= min_roi and s["pessimistic"].ev_net >= 0

    if not ok(0):
        return None
    lo, hi = 0.0, max(upper, 1.0)
    if ok(hi):
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def _breakeven_p10(base_probs: dict, prices: dict, low_price: float, price: float, costs: Costs) -> float | None:
    """基準情境下，PSA 10 機率要多少 EV 淨利才打平（其他等級按比例縮放）"""
    probs = normalize(base_probs)
    rest = {g: p for g, p in probs.items() if g != 10}
    rest_total = sum(rest.values())

    def net(p10: float) -> float:
        scale = (1 - p10) / rest_total if rest_total else 0
        pr = {g: p * scale for g, p in rest.items()}
        pr[10] = p10
        return scenario("x", pr, prices, low_price, price, costs).ev_net

    if net(0) >= 0:
        return 0.0
    if net(1) < 0:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if net(mid) < 0 else (lo, mid)
    return hi
