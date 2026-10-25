"""用 Claude 看圖，替每張 queue 照片預先標出瑕疵位置（給標註頁當初稿，人再確認）

用法：python tools/auto_label.py QUEUE_DIR [--model MODEL] [--effort low|medium|high]
  QUEUE_DIR/m<id>_<n>.jpg   fetch_photos.py 下載的照片
  QUEUE_DIR/m<id>.json      刊登資料（標題、品況，給模型當背景）
  → QUEUE_DIR/m<id>_<n>.label.json  {"side", "clean", "psa", "marks":[{"t","box","note"}], "model"}

需要 ANTHROPIC_API_KEY；每張照片一次呼叫，約 1,500～2,500 tokens。
框座標跟標註頁一樣是 0~1 的正規化座標 [x0, y0, x1, y1]。
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import anthropic

MARK_TYPES = ["edge", "corner", "spot", "scratch", "crease", "dent", "stain"]

PROMPT = """這是日本 Mercari 上一張寶可夢卡片的賣家照片，要送 PSA 鑑定前先看卡況。
刊登標題：{title}
賣家標示品況：{condition}

請只看「卡片本身」的實體瑕疵，忽略背景、反光、卡套、照片壓縮雜訊，也忽略印刷本來就有的白線、白框、文字。
找出每一處瑕疵並給出框（正規化座標，0~1，相對整張照片），類型：
- edge：邊緣白邊（卡邊的白色紙芯露出）
- corner：角尖白掉或磨圓
- spot：白點、小缺損
- scratch：刮痕（閃卡面上的細線）
- crease：摺痕、壓摺線
- dent：壓痕、凹陷
- stain：髒污、變色
框要緊貼瑕疵；同一條邊上連續的白邊給一個框即可。不確定的不要標，寧可少標。
side：這張是卡片正面 front、背面 back，還是看不出／不是卡片 other。
clean：這張照片裡看得到的部分完全沒有瑕疵時為 true。
psa：只在看得到整張卡（正面或背面全圖）時，估一個 PSA 分數 1~10；看不到整張就給 null。"""

SCHEMA = {
    "type": "object",
    "properties": {
        "side": {"type": "string", "enum": ["front", "back", "other"]},
        "clean": {"type": "boolean"},
        "psa": {"type": ["integer", "null"], "minimum": 1, "maximum": 10},
        "marks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "t": {"type": "string", "enum": MARK_TYPES},
                    "box": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}, "minItems": 4, "maxItems": 4},
                    "note": {"type": "string"},
                },
                "required": ["t", "box", "note"],
                "additionalProperties": False,
            },
        },
        "note": {"type": "string"},
    },
    "required": ["side", "clean", "psa", "marks", "note"],
    "additionalProperties": False,
}


def label_photo(client: anthropic.Anthropic, raw: bytes, listing: dict, model: str, effort: str) -> dict:
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(raw).decode()}},
        {"type": "text", "text": PROMPT.format(title=listing.get("title", ""), condition=listing.get("condition", ""))},
    ]
    resp = client.beta.messages.create(
        model=model,
        max_tokens=4000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": content}],
    )
    if resp.stop_reason == "refusal":
        return {"side": "other", "clean": False, "psa": None, "marks": [], "note": "模型拒絕判讀", "refused": True}
    text = next((b.text for b in resp.content if b.type == "text"), "{}")
    data = json.loads(text)
    # 框順序保險：x0<x1、y0<y1
    for m in data["marks"]:
        x0, y0, x1, y1 = m["box"]
        m["box"] = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
    data["usage"] = {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("queue", type=Path)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium")
    a = ap.parse_args()
    client = anthropic.Anthropic()
    photos = sorted(a.queue.glob("m*_*.jpg"))
    done = skipped = 0
    tok_in = tok_out = 0
    for p in photos:
        out = p.with_suffix(".label.json")
        if out.exists():
            skipped += 1
            continue
        lid = p.stem.split("_")[0]
        lj = a.queue / f"{lid}.json"
        listing = json.loads(lj.read_text()) if lj.exists() else {}
        try:
            data = label_photo(client, p.read_bytes(), listing, a.model, a.effort)
        except anthropic.APIStatusError as e:
            print(f"{p.name}: API 錯誤 {e.status_code} {e.message}", file=sys.stderr)
            continue
        data["model"] = a.model
        out.write_text(json.dumps(data, ensure_ascii=False))
        tok_in += data.get("usage", {}).get("in", 0)
        tok_out += data.get("usage", {}).get("out", 0)
        done += 1
        print(f"{p.name}: {data['side']} marks={len(data['marks'])} clean={data['clean']} psa={data['psa']}")
    print(f"標了 {done} 張（略過已有 {skipped}），tokens in {tok_in} / out {tok_out}")


if __name__ == "__main__":
    main()
