# 🇨🇭 Swiss Health Insurance — Premium Region Classification

**HEC Lausanne | Master in Business Analytics | Machine Learning Course 2026**  
**Prof. Arnold Vialfont**

---

## Research Question

> Can a Machine Learning model predict the official premium region (1, 2 or 3) of a Swiss commune based solely on its socio-demographic profile — and detect communes that are financially penalized by a politically frozen classification system?

Switzerland's health insurance premium regions have been **frozen since 2004** (EPFZ study) and further locked by parliamentary motion 18.3713. Our model learns from current socio-demographic reality and identifies communes whose official classification no longer reflects their actual profile.

---

## Project Structure

```
ML_Project/
│
├── notebooks/
│   └── swiss_health_insurance_ML_v3.ipynb   ← main notebook
│
├── data/                                     ← raw data (not tracked by Git)
│   ├── Anhang_EDI_Ver__über_die_PrReg_Mut_2026.xlsx
│   ├── praemienregionen-ab-2025__1_.xlsx
│   ├── data.xlsx
│   ├── Prämien_CH_2025.xlsx
│   └── Prämien_CH_2026.xlsx
│
├── outputs/                                  ← generated automatically
│   ├── swiss_communes_ml_ready.csv           ← final ML-ready dataset
│   ├── eda_financial_stakes.png
│   ├── eda_berne_gap.png
│   ├── eda_sociodemographic.png
│   └── eda_correlation.png
│
├── requirements.txt
└── README.md
```

---

## Data Sources

| File | Source | Content | Role in ML |
|------|--------|---------|------------|
| `Anhang_EDI_Ver_2026.xlsx` | OFSP (bag.admin.ch) | Official commune → premium region mapping | **Target Y** |
| `data.xlsx` | OFS (Swiss Stats Map Explorer) | Socio-demographic stats per commune, 2024 | **Features X** |
| `Prämien_CH_2025.xlsx` | opendata.swiss | Premium prices by insurer/canton/region, 2025 | **EDA enrichment** |
| `Prämien_CH_2026.xlsx` | opendata.swiss | Premium prices by insurer/canton/region, 2026 | **EDA enrichment** |

> ⚠️ Raw data files are **not tracked by Git** (too large / proprietary). Download them from the sources above and place them in the `data/` folder before running the notebook.

---

## Key Design Decision — join_key

In the premium files, region codes are `PR-REG CH0/1/2/3`:
- `CH0` = **single-region canton** (GE, AG, BS, etc.) → one uniform premium for the whole canton
- `CH1/2/3` = regions 1/2/3 within a **multi-region canton** (BE, ZH, VD, etc.)

We use `canton + '_H' + region_code_suffix` as a composite join key (e.g. `GE_H0`, `BE_H1`, `ZH_H3`). This ensures Geneva (single region, `GE_H0`) and Berne zone 1 (`BE_H1`) are **never merged** into the same market context, which would bias all premium-based features.

---

## ML Pipeline

### Data
- **~1,500 communes** after filtering and fusion handling
- **~23 features** across two groups:
  - OFS socio-demographic (population density, household composition, social assistance rate, etc.)
  - Premium market features (avg/median/std/p10/p90/IQR premium, n_insurers, YoY increase)
- **Target**: `premium_region` ∈ {1, 2, 3} — class distribution: 332 / 824 / 347

### Steps
1. **Data cleaning** — commune fusions (mutations), column renaming, median imputation
2. **EDA** — financial stakes, intra-canton gaps, socio-demographic profiles, correlation matrix
3. **Unsupervised** — PCA (dimensionality reduction) + KMeans k=3 (compare to official regions)
4. **Supervised** — Logistic Regression → Random Forest → XGBoost, tuned with GridSearchCV + 5-fold CV
5. **Interpretability** — SHAP values + financial impact analysis of misclassified communes

### Metrics
- **Cohen's Kappa** (primary — handles multiclass imbalance)
- Macro-F1
- Accuracy
- Confusion matrix (3×3)

---

## Key EDA Findings

**Financial stakes (Berne example)**  
Within the same canton, the same adult with the same coverage pays CHF 657/month in Region 1 vs CHF 554 in Region 3 — a gap of **CHF 103/month = CHF 1,238/year** for identical insurance.

**Premium increases 2025→2026**  
Average +3.5% on standard adult profile. Tessin leads at +5.8%, Vaud lowest at +2.4%.

**Strongest predictors of premium_region** (from correlation matrix)  
`avg_premium_2026` (−0.67), `p90_premium_2026` (−0.69), `pct_large_hh` (+0.34), `social_assistance_rate` (+0.14).

**Key multicollinearity finding**  
Premium features (avg, median, p10, p90) are correlated at 0.98–1.00 → PCA will handle this redundancy in the unsupervised step.

---

## Setup & Installation

### Requirements
- Python 3.9+
- Anaconda recommended

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the notebook
```bash
cd notebooks
jupyter notebook swiss_health_insurance_ML_v3.ipynb
```

Or open directly in VS Code with the Jupyter extension.

### Update paths
In cell **0 — Setup & Paths**, set your local data folder:
```python
BASE = r'C:\your\path\to\ML_Project\data' + os.sep
```

---

## Requirements

```
pandas
openpyxl
scikit-learn
matplotlib
seaborn
numpy
jupyter
shap
xgboost
```

---

## Authors

- Mattia — HEC Lausanne Master Business Analytics

---

## References

- OFSP (2025). *Ordonnance du DFI sur les régions de primes*, valid 01.01.2026. bag.admin.ch
- OFS (2024). *Swiss Stats Map Explorer — Communes 01.01.2024*. bfs.admin.ch  
- RTS (2025). *Les primes maladie grimperont de 4,4% en 2026*. rts.ch
- Fiche d'information OFSP (2024). *Régions de primes dans l'assurance obligatoire des soins (AOS)*.
