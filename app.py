"""
AQI Intelligence Dashboard & Predictor
========================================
A single-file Streamlit application that:
  1. Loads and cleans India air-quality data (city_day.csv style)
  2. Surfaces KPIs, trends, pollutant drivers, and risk/opportunity insights
     (the Business Intelligence framework: KPI -> Trend -> Driver -> Risk -> Action)
  3. Trains a regression model (predict exact AQI value) and a classification
     model (predict AQI category / bucket) on the cleaned data
  4. Lets the user enter pollutant readings and get a live AQI prediction

Dataset: "Air Quality Data in India (2015-2020)" by Rohan Rao (Kaggle)
Link: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
Place city_day.csv in the same folder as this file before running.

Run with:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "city_day.csv"
POLLUTANT_COLS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

st.set_page_config(page_title="AQI Intelligence Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# 1. DATA LOADING + CLEANING
# ---------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Keep only columns this app needs (be tolerant of extra pollutant columns
    # the full Kaggle file has, like NH3/Benzene/Toluene/Xylene)
    keep_cols = ["City", "Date"] + [c for c in POLLUTANT_COLS if c in df.columns] \
        + [c for c in ["AQI", "AQI_Bucket"] if c in df.columns]
    df = df[keep_cols].copy()

    # Drop rows with no AQI at all -- can't train or report on those
    df = df.dropna(subset=["AQI"])

    # Fill missing pollutant readings per-city with that city's median
    # (pollutant levels vary a lot by city, so a global median would distort things)
    for col in POLLUTANT_COLS:
        if col in df.columns:
            df[col] = df.groupby("City")[col].transform(
                lambda s: s.fillna(s.median())
            )
            df[col] = df[col].fillna(df[col].median())  # any city-wide gaps left

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%b")

    return df.reset_index(drop=True)


@st.cache_resource
def train_models(df: pd.DataFrame):
    feature_cols = [c for c in POLLUTANT_COLS if c in df.columns]
    model_df = df.dropna(subset=feature_cols + ["AQI"])

    X = model_df[feature_cols]
    y_reg = model_df["AQI"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_train)
    reg_metrics = {
        "R2": round(r2_score(y_test, reg.predict(X_test)), 3),
        "MAE": round(mean_absolute_error(y_test, reg.predict(X_test)), 2),
    }

    clf_metrics, clf, label_encoder = None, None, None
    if "AQI_Bucket" in model_df.columns:
        y_clf_raw = model_df["AQI_Bucket"]
        label_encoder = LabelEncoder()
        y_clf = label_encoder.fit_transform(y_clf_raw)

        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
        )
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(Xc_train, yc_train)
        clf_metrics = {
            "Accuracy": round(accuracy_score(yc_test, clf.predict(Xc_test)), 3),
            "F1 (macro)": round(f1_score(yc_test, clf.predict(Xc_test), average="macro"), 3),
        }

    return {
        "feature_cols": feature_cols,
        "reg": reg, "reg_metrics": reg_metrics,
        "clf": clf, "clf_metrics": clf_metrics, "label_encoder": label_encoder,
    }


# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------
st.title("🌫️ AQI Intelligence Dashboard")
st.caption("Business-intelligence view of India's air quality: KPIs, trends, drivers, risks, and a live AQI predictor.")

df = load_and_clean_data(DATA_PATH)

if df is None:
    st.error(
        f"Couldn't find `{DATA_PATH}`. Download **city_day.csv** from the "
        "[Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) "
        "Kaggle dataset and place it next to this script."
    )
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
cities = sorted(df["City"].unique())
selected_cities = st.sidebar.multiselect("City", cities, default=cities[:5])
year_range = st.sidebar.slider(
    "Year range", int(df["Year"].min()), int(df["Year"].max()),
    (int(df["Year"].min()), int(df["Year"].max()))
)

filtered = df[
    df["City"].isin(selected_cities)
    & df["Year"].between(*year_range)
] if selected_cities else df.copy()

tab_overview, tab_trends, tab_drivers, tab_risk, tab_predict = st.tabs(
    ["📊 Executive Overview", "📈 Trends", "🔍 Drivers", "⚠️ Risk & Action", "🤖 Predict AQI"]
)

# --- TAB 1: KPIs (executive overview) --------------------------------------
with tab_overview:
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average AQI", f"{filtered['AQI'].mean():.0f}")
    hazardous_pct = (
        filtered["AQI_Bucket"].isin(["Poor", "Very Poor", "Severe"]).mean() * 100
        if "AQI_Bucket" in filtered.columns else np.nan
    )
    col2.metric("% Poor-or-worse Days", f"{hazardous_pct:.1f}%")
    worst_city = filtered.groupby("City")["AQI"].mean().idxmax() if len(filtered) else "-"
    col3.metric("Worst-Avg City", worst_city)
    best_city = filtered.groupby("City")["AQI"].mean().idxmin() if len(filtered) else "-"
    col4.metric("Best-Avg City", best_city)

    st.divider()
    st.subheader("City-wise Average AQI")
    city_avg = filtered.groupby("City")["AQI"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(city_avg, x="City", y="AQI", color="AQI", color_continuous_scale="RdYlGn_r")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: Trends -----------------------------------------------------------
with tab_trends:
    st.subheader("AQI Trend Over Time")
    trend = filtered.groupby(["Date"])["AQI"].mean().reset_index()
    fig = px.line(trend, x="Date", y="AQI")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Seasonal Pattern (Average AQI by Month)")
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly = filtered.groupby("Month_Name")["AQI"].mean().reindex(month_order).reset_index()
    fig2 = px.bar(monthly, x="Month_Name", y="AQI")
    st.plotly_chart(fig2, use_container_width=True)
    st.info(
        "Winter months typically show higher AQI (still air, temperature inversion, "
        "crop burning), while monsoon months show relief from rain washing out particulates."
    )

# --- TAB 3: Drivers -----------------------------------------------------------
with tab_drivers:
    st.subheader("Which Pollutants Drive AQI the Most?")
    available = [c for c in POLLUTANT_COLS if c in filtered.columns]
    corr = filtered[available + ["AQI"]].corr()["AQI"].drop("AQI").sort_values(ascending=False)
    fig = px.bar(corr, orientation="h", labels={"value": "Correlation with AQI", "index": "Pollutant"})
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        f"**{corr.index[0]}** shows the strongest correlation with AQI in the selected filter — "
        "this is the pollutant most worth targeting with interventions."
    )

# --- TAB 4: Risk & Action -----------------------------------------------------
with tab_risk:
    st.subheader("Risk: Cities With Most Hazardous Days")
    if "AQI_Bucket" in filtered.columns:
        hazardous = filtered[filtered["AQI_Bucket"].isin(["Poor", "Very Poor", "Severe"])]
        risk_city = hazardous["City"].value_counts().reset_index()
        risk_city.columns = ["City", "Hazardous_Days"]
        fig = px.bar(risk_city, x="City", y="Hazardous_Days", color="Hazardous_Days",
                     color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recommended Actions")
    st.markdown(
        """
        - **Highest-risk cities** need priority monitoring and public health advisories during winter months.
        - The **top driver pollutant** (see Drivers tab) should be the focus of emission-control policy
          (e.g. vehicle/industrial restrictions, construction dust control).
        - **Opportunity:** cities with consistently low AQI can be studied for practices worth replicating elsewhere.
        """
    )

# --- TAB 5: Predict -----------------------------------------------------------
with tab_predict:
    st.subheader("Predict AQI From Pollutant Readings")
    models = train_models(df)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Model performance (on held-out test data)**")
        st.write("Regression (exact AQI value):", models["reg_metrics"])
        if models["clf_metrics"]:
            st.write("Classification (AQI category):", models["clf_metrics"])

    with c2:
        st.markdown("**Enter pollutant readings**")
        inputs = {}
        for col in models["feature_cols"]:
            default_val = float(df[col].median())
            inputs[col] = st.number_input(col, value=default_val, min_value=0.0)

        if st.button("Predict AQI", type="primary"):
            X_input = pd.DataFrame([inputs])[models["feature_cols"]]
            pred_aqi = models["reg"].predict(X_input)[0]
            st.success(f"Predicted AQI value: **{pred_aqi:.1f}**")

            if models["clf"] is not None:
                pred_class_idx = models["clf"].predict(X_input)[0]
                pred_class = models["label_encoder"].inverse_transform([pred_class_idx])[0]
                st.success(f"Predicted AQI category: **{pred_class}**")

st.divider()
st.caption("Built for the BharatCares / Data Decoded AI & ML masterclass final project.")
