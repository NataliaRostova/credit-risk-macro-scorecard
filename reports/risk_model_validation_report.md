# 商业银行信用风险评分卡与 IFRS 9 宏观压力测试验证报告

## 一、模型验证核心指标（Model Validation Scorecard）

| 验证维度 | 指标 | 实测结果 | 银行业风控标准参考 |
| :--- | :--- | :---: | :--- |
| 区分度（Discrimination） | Holdout Test ROC-AUC | **0.7355** | > 0.75（优秀） |
| 跨时间泛化（OOT Generalization） | OOT ROC-AUC / Gini | **0.7362 / 0.4724** | AUC > 0.72 |
| 风险分离度 | Test KS / OOT KS | **0.4121 / 0.4241** | KS > 0.35（强分离） |
| 群体稳定性 | OOT PSI | **0.0318** | < 0.10（高度稳定） |
| 尾部风险捕获 | Top 20% 高风险组违约捕获率 | **47.2%** | > 50% |

---

## 二、特征信息价值（IV）排序

| 财务 / 信用特征 | IV 值 |
| :--- | :---: |
| `debt_to_asset_pct` | 0.4531 |
| `credit_utilization_pct` | 0.3151 |
| `ocf_to_debt` | 0.3088 |
| `interest_coverage` | 0.1281 |
| `roe_pct` | 0.0912 |
| `current_ratio` | 0.0811 |
| `delinquency_count_12m` | 0.0625 |
| `revenue_yoy_pct` | 0.0298 |

---

## 三、IFRS 9 宏观压力测试（ECL = PD × LGD × EAD）

| 宏观压力情景 | 组合平均 PD | 假设 LGD | 总敞口 EAD (百万元) | 预期信用损失 ECL (百万元) | 损失率 (bps) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline (Current) | 5.06% | 45.0% | 15,001.5 | **344.62** | 229.7 bps |
| Adverse (+100bps / -8% Rev) | 6.77% | 50.0% | 15,001.5 | **516.60** | 344.4 bps |
| Severely Adverse (+200bps / -18% Rev) | 9.31% | 55.0% | 15,001.5 | **785.69** | 523.7 bps |
