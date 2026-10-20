# Merryan

Mercari 新上架監控（GitHub Actions → Telegram），加上裸卡買來送 PSA 鑑定的期望值評估。

## 鑑定期望值（EV）評估

品項在 `config.yaml` 加上 `card: <cards.yaml 的卡種>`，新刊登通知就會附上判斷：

```
💰 建議買進 | ルギア e-P 047
¥60,000（NT$12,444） · 目立った傷や汚れなし
期望淨利 基準 NT$12,242（ROI 81%）/ 悲觀 NT$4,756 / 樂觀 NT$21,345
鑑定增量 NT$13,154 · PSA10 機率 10%（打平需 0%）
最高出價 ¥82,484
依據：
- 商品狀態「目立った傷や汚れなし」→ 以基準為中心
- [heuristic] 量測置中 左右 50/50、上下 50/50（信心 100%）
```

流程：

1. 打開商品頁，抓價格、商品狀態、描述、所有照片。
2. 估分數機率（`grade_estimator.py`）：
   - 商品狀態選起點：1〜3 以「基準」為中心，4〜5 以「悲觀」為中心，6 跳過。
   - 描述瑕疵字眼（白欠け、初期傷、折れ…）給上限或往下移，「傷なし」這類否定句不算。
   - 照片分析（`image_analysis.py`）：置中依 PSA 標準給上限（60/40 以內才有 10），邊角白邊、表面、折痕也會壓上限。
   - 照片少於 2 張或沒有背面，往悲觀方向混合。
   - 賣家的「美品」等正面字眼不加分。
3. 算 EV（`ev_calc.py`）：總成本含代購、國際運費、匯損、PSA 鑑定費（依申報價值級距）、往返運保、賣出手續費。
4. 判斷：基準 ROI ≥ 25%、悲觀不虧、鑑定增量 > 0 才是「建議買進」；基準達標但悲觀會虧是「可考慮」。另外算出符合門檻的最高出價，以及 PSA 10 機率要多少才打平。

通知只代表「值得點進去驗卡」，下手前仍要自己看高解析照片。

### 照片分析器

- `heuristic`：純像素量測，量正面黃框置中、背面藍底邊角白點。照片要大致正放、卡片輪廓清楚才會有高信心；斜拍或卡套反光會被判低信心而不採用。
- `claude`：設定 GitHub Secret `ANTHROPIC_API_KEY` 後自動啟用，用 Claude 看照片判讀置中、邊角、表面、折痕、仿品疑慮（會產生 API 費用）。
- 要接其他模型：寫一個有 `name` 與 `analyze(images, listing) -> Observation` 的類別，加進 `build_analyzers()`。

### 要填的數字（`cards.yaml`）

計價幣別是新台幣（Mercari 價格當日圓換算）。匯率用臺銀即期賣出牌告；PSA 鑑定費為官網價加 20%；運費暫不計。代購費也暫不計。標「範例」的仍是假設值：賣出手續費、機率組。最高出價以日圓顯示，直接對照 Mercari 標價。

### 單張手動評估

```bash
python ev_check.py https://jp.mercari.com/item/m12345678901 --card lugia-e-p-047
python ev_check.py --card lugia-e-p-047 --price 80000 --condition 3 --photos front.jpg back.jpg --log ev_log.csv
```

`--log` 會把預估寫進 CSV，鑑定回來後補 `actual_grade`，累積 20〜30 張再校準機率組。

### 測試

```bash
pip install -r requirements.txt pytest
python -m pytest tests
```
