# 產品說明

A4T 是一套針對加密貨幣交易所的詐欺偵測系統，透過分析使用者在多種交易類型中的行為模式來識別詐欺帳號。

涵蓋的交易類型：
- TWD 轉帳（`twd_transfer`）
- 加密貨幣轉帳（`crypto_transfer`）
- USDT/TWD 交易（`usdt_twd_trading`）
- USDT 兌換（`usdt_swap`）

## 模型架構

系統採用集成（ensemble）方式，結合三個模型：
- LightGBM — 處理表格式特徵（tabular features）
- GraphSAGE — 圖神經網路（graph neural network），捕捉使用者間的關聯
- Stacking 集成 — 結合上述兩個模型的預測結果

## 核心領域概念

- 使用者標籤：詐欺為 `1`，正常為 `0`
- 原始資料格式：JSONL（每行一筆 JSON）
- 特徵類型：行為特徵、時序特徵、圖特徵、社群偵測特徵
- 資料洩漏（data leakage）防範至關重要：社群特徵與標籤相關特徵，必須只使用當前 CV fold 的訓練集標籤計算
