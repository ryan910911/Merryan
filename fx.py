"""匯率：每次執行抓當天匯率換成台幣，抓不到就用 cards.yaml 的 fx 固定值

cards.yaml 的行情多半是 PriceCharting 的美金價（"2861 USD"），程式內部一律換成台幣計算與顯示。
即時匯率來源：open.er-api.com（ExchangeRate-API 免費版，每天更新一次的中價）。
臺銀「即期賣出」比中價貴一點，用 cards.yaml 的 fx_live.markup（預設 0.4%）補上。
"""
from __future__ import annotations

import os

import requests

SOURCE = "https://open.er-api.com/v6/latest/TWD"


def fetch_twd_rates(currencies, timeout: float = 10.0) -> tuple[dict, str]:
    """回傳 ({幣別: 1 單位 = 多少台幣}, 資料時間)"""
    r = requests.get(SOURCE, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if data.get("result") != "success":
        raise RuntimeError(f"匯率 API 回應異常：{data.get('error-type') or data.get('result')}")
    per_twd = data["rates"]  # 1 TWD = x 外幣
    out = {}
    for cur in currencies:
        if cur == "TWD":
            continue
        v = float(per_twd[cur])
        if v <= 0:
            raise RuntimeError(f"{cur} 匯率異常：{v}")
        out[cur] = 1 / v
    return out, data.get("time_last_update_utc", "")


def apply_live_fx(cfg: dict) -> dict:
    """把 cfg["fx"] 換成當天匯率（就地修改並回傳 cfg）。

    關閉方式：cards.yaml 設 fx_live.enabled: false，或環境變數 MERRYAN_FX_LIVE=0。
    抓到的匯率若跟固定值差超過 max_change（預設 20%），視為資料有誤，沿用固定值。
    """
    fx = cfg.setdefault("fx", {"TWD": 1})
    opts = cfg.get("fx_live") or {}
    cfg["fx_source"] = "cards.yaml 固定匯率"
    if os.environ.get("MERRYAN_FX_LIVE", "1") == "0" or opts.get("enabled", True) is False:
        return cfg
    markup = float(opts.get("markup", 0.004))
    max_change = float(opts.get("max_change", 0.2))
    try:
        live, when = fetch_twd_rates([c for c in fx if c != "TWD"])
    except Exception as e:  # 沒網路或 API 掛了：用固定值
        print("抓即時匯率失敗，改用 cards.yaml 固定匯率：", e)
        return cfg
    for cur, rate in live.items():
        rate *= 1 + markup
        old = float(fx[cur])
        if old and abs(rate / old - 1) > max_change:
            print(f"{cur} 即時匯率 {rate:.4f} 與固定值 {old} 差太多，沿用固定值")
            continue
        fx[cur] = round(rate, 6)
    cfg["fx_source"] = f"即時匯率 {when[5:16] if when else '今日'}，中價 +{markup:.1%}"
    return cfg


def fx_summary(cfg: dict) -> str:
    fx = cfg.get("fx") or {}
    parts = [f"1 {c} = {float(v):.4g} 元" for c, v in fx.items() if c != "TWD"]
    return "匯率：" + "、".join(parts) + f"（{cfg.get('fx_source', 'cards.yaml 固定匯率')}）"
