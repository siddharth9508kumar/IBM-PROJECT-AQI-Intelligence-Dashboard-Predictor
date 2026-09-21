# 🌫️ AQI Intelligence Dashboard & Predictor

An interactive Streamlit dashboard and machine learning app that analyzes India's historical air quality data, highlights the pollutants, cities and seasons that matter most, and predicts the Air Quality Index (AQI) from pollutant readings.

The project follows a Business Intelligence framework: **KPI → Trend → Driver → Risk → Action**.

📄 **Project report:** [Google Doc](https://docs.google.com/document/d/1R-sRK9Ivi4dESMkYF7_B0V_1WEm6N3eLKX9QVspJNCU/edit?usp=sharing)

---

## Table of Contents

- [Features](#features)
- [Dashboard Preview](#dashboard-preview)
- [Model Performance](#model-performance)
- [Key Findings](#key-findings)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Using the App](#using-the-app)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)
- [Tech Stack](#tech-stack)
- [Acknowledgements](#acknowledgements)

---

## Features

The app is a single page with sidebar filters (city, year range) and five tabs:

| Tab | Question it answers | What you see |
| --- | --- | --- |
| 📊 **Executive Overview** | How bad is it, and where? | Average AQI, % of Poor-or-worse days, best and worst cities, city-wise AQI bar chart |
| 📈 **Trends** | How does AQI change over time? | AQI over time and average AQI by month |
| 🔍 **Drivers** | Which pollutant matters most? | Correlation of PM2.5, PM10, NO2, SO2, CO and O3 with AQI |
| ⚠️ **Risk & Action** | Who is most at risk, and what should be done? | Cities ranked by hazardous days, plus recommended actions |
| 🤖 **Predict AQI** | What AQI do these readings imply? | Enter pollutant values and get a predicted AQI and category, with live model metrics |

**Two models** are trained on startup (Random Forest, 200 trees each):
- **Regression:** predicts the exact AQI value.
- **Classification:** predicts the AQI category (Good / Satisfactory / Moderate / Poor / Very Poor / Severe).

---

## Dashboard Preview

> Charts below are generated from the cleaned dataset (all 26 cities, 2015–2020) and mirror what each tab shows. The live app opens with the first five cities selected in the sidebar.

**Executive Overview: city-wise average AQI**

![City-wise average AQI](docs/images/overview.png)

**Trends: AQI over time and seasonal pattern**

![Trends](docs/images/trends.png)

**Drivers: correlation of each pollutant with AQI**

![Drivers](docs/images/drivers.png)

**Risk: cities with the most hazardous days**

![Risk](docs/images/risk.png)

**Prediction: model fit and feature importance**

![Prediction](docs/images/predict.png)

---

## Model Performance

Evaluated on a held-out 20% test set (19,880 training rows, 4,970 test rows, `random_state=42`). The same numbers appear live in the **Predict AQI** tab and may differ slightly across library versions.

| Model | Metric | Score |
| --- | --- | --- |
| Regression (exact AQI) | R² | **0.905** |
| | MAE | **21.93** AQI points |
| Classification (AQI category) | Accuracy | **0.806** |
| | F1 (macro) | **0.782** |

Of the classifier's predictions, 99.4% fall within one category of the true value, so its errors are almost always between neighbouring bands. See [Known Limitations](#known-limitations) for how far these scores can be trusted.

---

## Key Findings

*Based on all 26 cities, 24,850 city-days, 2015–2020.*

- **Average AQI is 166.5**, and **26.0%** of city-days are Poor, Very Poor or Severe.
- **Hazardous air is concentrated:** Delhi, Ahmedabad, Lucknow, Patna and Gurugram account for **75.3%** of all hazardous days.
- **Winter is the danger season:** 65.5% of hazardous days fall between October and February. In Delhi, 97.1% of days in November–January are Poor or worse.
- **PM2.5 is the key pollutant** once Ahmedabad's unusual CO readings are set aside (correlation 0.88 with AQI). With that city included, the app ranks CO first (0.68). See [Known Limitations](#known-limitations).
- **Cleanest cities:** Aizawl (34.8), Shillong (53.8) and Coimbatore (73.0) average the lowest AQI.

---

## Dataset

**Air Quality Data in India (2015–2020)** by Rohan Rao, on Kaggle:
<https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india>

The app uses `city_day.csv`, which contains daily readings for 26 cities from 1 Jan 2015 to 1 Jul 2020 (29,531 rows).

| Column | Description |
| --- | --- |
| `City` | Monitoring city |
| `Date` | Date of the reading |
| `PM2.5`, `PM10` | Particulate matter concentrations |
| `NO2`, `SO2`, `CO`, `O3` | Gas pollutant concentrations |
| `AQI` | Air Quality Index (regression target) |
| `AQI_Bucket` | AQI category (classification target) |

The file also contains NO, NOx, NH3, Benzene, Toluene and Xylene columns, which the app ignores. A copy of `city_day.csv` is included in this repository. For license terms, see the dataset's Kaggle page.

---

## Getting Started

### Prerequisites

- Python 3.10 or newer (tested on 3.12)
- `pip`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/siddharth9508kumar/IBM-PROJECT-AQI-Intelligence-Dashboard-Predictor.git
cd IBM-PROJECT-AQI-Intelligence-Dashboard-Predictor

# 2. (Recommended) create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Streamlit prints a local URL (usually <http://localhost:8501>). Open it in your browser.

> `city_day.csv` must be in the same folder as `app.py`. If it is missing, the app shows an error with the download link.

---

## Using the App

1. **Filter the data** in the sidebar: choose cities and a year range. The app opens with the first five cities (alphabetically) selected, so add more cities, or select all of them, to see the national picture.
2. **Explore the tabs** from Executive Overview through Risk & Action.
3. **Predict AQI:** open the **Predict AQI** tab, enter values for PM2.5, PM10, NO2, SO2, CO and O3 (defaults are dataset medians) and click **Predict AQI**. You get a predicted AQI value and a predicted category.

With the default inputs, the app returns an AQI of about **115.5** (category **Moderate**).

---

## Project Structure

```
.
├── app.py                  # Data cleaning, dashboard, and both ML models (single file)
├── city_day.csv            # Dataset (Kaggle: Air Quality Data in India)
├── requirements.txt        # Python dependencies
├── AQI_Project_Report.docx # Project report
├── docs/
│   └── images/             # Charts used in this README
└── README.md               # This file
```

---

## How It Works

### 1. Data cleaning (`load_and_clean_data`)

1. Parse the `Date` column and keep only the columns the app needs.
2. Drop rows with no AQI value (4,681 of 29,531 rows).
3. Fill missing pollutant readings with **each city's own median**, since pollution levels vary widely by location, with a global median as a fallback.
4. Drop exact duplicate rows (none in this file).
5. Add `Year`, `Month` and `Month_Name` columns for the dashboard.

### 2. Modeling (`train_models`)

| Step | Regression | Classification |
| --- | --- | --- |
| Features | PM2.5, PM10, NO2, SO2, CO, O3 | same |
| Target | `AQI` | `AQI_Bucket` (label-encoded) |
| Model | `RandomForestRegressor` (200 trees) | `RandomForestClassifier` (200 trees) |
| Split | 80/20 | 80/20, stratified by category |
| Metrics | R², MAE | Accuracy, macro F1 |

Data loading is cached with `st.cache_data`, and the trained models with `st.cache_resource`, so the app stays responsive while you change filters.

### 3. Business-intelligence logic

- **KPIs:** mean AQI, share of `Poor`, `Very Poor` and `Severe` days, and the best and worst cities by mean AQI.
- **Drivers:** Pearson correlation of each pollutant with AQI for the selected filters.
- **Risk:** count of hazardous days per city.

---

## Known Limitations

Full details are in Section 9 of the [project report](AQI_Project_Report.docx).

- **Ahmedabad's readings look anomalous.** Its median CO is about 18× that of the other cities, and its AQI reaches 2,049 (the Severe band is 401–500). This makes Ahmedabad the "worst" city and CO the top driver when all cities are selected. The cause hasn't been verified and should be checked at the source.
- **The predictor estimates AQI from same-day readings; it is not a forecast.** AQI is calculated from pollutant concentrations, so high scores partly reflect the model re-learning that calculation. The app reads a static CSV and does not use live data.
- **Evaluation is optimistic.** Scores come from a random row split. On entirely unseen cities, R² ranges from about 0.80 (Delhi, Mumbai) to negative (Ahmedabad).
- **Imputation:** 35.6% of rows had at least one pollutant filled with a city median, which flattens seasonal variation in those values.
- **Trends are not like-for-like:** the number of reporting cities changes across years, and 2020 covers January–July only (including the COVID-19 lockdown).
- **Risk ranking uses raw day counts,** which favours cities with longer records.

---

## Future Work

- [ ] Validate and correct the Ahmedabad CO/AQI data
- [ ] Add next-day AQI forecasting using lagged and rolling features (and weather data)
- [ ] Evaluate on time-based and leave-one-city-out splits
- [ ] Show hazardous days as a percentage of each city's records
- [ ] Use a constant-city panel for year-over-year trends
- [ ] Default the sidebar to all cities
- [ ] Add tests, save the trained model, and pin dependency versions

---

## Tech Stack

| Layer | Tool |
| --- | --- |
| Language | Python |
| Data handling | pandas, numpy |
| Machine learning | scikit-learn (Random Forest regression + classification) |
| Dashboard | Streamlit |
| Visualization | Plotly |

---

## Acknowledgements

- Dataset: [Air Quality Data in India (2015–2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) by Rohan Rao, on Kaggle.
- Built as the final project for the **BharatCares / Data Decoded AI & ML Masterclass**.

**Author:** [@siddharth9508kumar](https://github.com/siddharth9508kumar)
