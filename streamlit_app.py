"""Streamlit dashboard: influences of climatic patterns on food supply in Kenya."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.charts import (
    correlation_heatmap,
    price_change_distribution,
    price_trend,
    production_index,
    scatter_pair,
)
from src.data import PCT_SUFFIX, PRICE_COLUMNS, load_data
from src.models import (
    PROPHET_AVAILABLE,
    forecast_gpr,
    forecast_prophet,
    forecast_random_forest,
)

st.set_page_config(
    page_title="Climate & Food Supply Dashboard",
    page_icon="\U0001F30D",
    layout="wide",
)


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_data()


DATA = get_data()


@st.cache_data(show_spinner="Training model...")
def run_forecast(model_type: str, target: str, features: tuple[str, ...], horizon: int, method: str):
    if model_type == "Gaussian Process Regression":
        result = forecast_gpr(DATA, target, features, horizon, method)
    elif model_type == "Random Forest Regressor":
        result = forecast_random_forest(DATA, target, features, horizon, method)
    else:
        result = forecast_prophet(DATA, target, features, horizon, method)
    return result.metrics, result.forecast, result.residuals


def fmt(value: float) -> str:
    return f"{value:,.2f}" if value is not None and np.isfinite(value) else "n/a"


def main() -> None:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:6px;">
          <span style="font-size:2.6rem;">\U0001F331</span>
          <h1 style="color:#2e8b57;margin:0;">Influences of Climatic Patterns on Food Supply in Kenya</h1>
        </div>
        <p style="color:#555;margin-top:0;">Explore historical crop prices and climate variables, then forecast the next few years.</p>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    section = st.sidebar.radio("**Dashboard section**", ["Exploratory analysis", "Model predictions"])

    if section == "Exploratory analysis":
        analysis = st.sidebar.radio(
            "**Analysis**",
            [
                "Price trends",
                "Food production index",
                "Correlation",
                "Pairwise scatter",
                "Price-change distribution",
            ],
        )

        if analysis == "Price trends":
            crops = st.sidebar.multiselect("Crops", PRICE_COLUMNS, default=PRICE_COLUMNS)
            if crops:
                st.plotly_chart(price_trend(DATA, crops), use_container_width=True)
            else:
                st.info("Select at least one crop.")

        elif analysis == "Food production index":
            st.plotly_chart(production_index(DATA), use_container_width=True)

        elif analysis == "Correlation":
            st.plotly_chart(correlation_heatmap(DATA), use_container_width=True)

        elif analysis == "Pairwise scatter":
            x = st.sidebar.selectbox("X variable", DATA.columns)
            y = st.sidebar.selectbox("Y variable", DATA.columns, index=min(1, len(DATA.columns) - 1))
            if x == y:
                st.info("Choose two different variables.")
            else:
                st.plotly_chart(scatter_pair(DATA, x, y), use_container_width=True)

        else:
            crops = st.sidebar.multiselect("Crops", PRICE_COLUMNS, default=PRICE_COLUMNS)
            if crops:
                st.plotly_chart(price_change_distribution(DATA, crops), use_container_width=True)
            else:
                st.info("Select at least one crop.")

    else:
        target = st.sidebar.selectbox("**Target crop price**", PRICE_COLUMNS)
        target_pct = f"{target}{PCT_SUFFIX}"
        numeric_columns = DATA.select_dtypes(include="number").columns.tolist()
        feature_options = [c for c in numeric_columns if c not in {*PRICE_COLUMNS, "Year", target_pct}]
        features = st.sidebar.multiselect("**Feature columns**", feature_options, default=feature_options)
        model_type = st.sidebar.selectbox(
            "**Model**",
            ["Gaussian Process Regression", "Random Forest Regressor", "Prophet"],
        )
        horizon = st.sidebar.slider("Forecast horizon (years)", 1, 10, 5)
        method = st.sidebar.radio(
            "Future driver scenario",
            ["trend", "constant"],
            horizontal=True,
            help="How future temperature, rainfall and other drivers are projected.",
        )

        st.subheader(f"Forecast: {target}")

        if model_type == "Prophet" and not PROPHET_AVAILABLE:
            st.warning("Prophet is not installed. Run `pip install -r requirements.txt` to enable it.")
            st.stop()
        if not features:
            st.warning("Select at least one feature column.")
            st.stop()

        try:
            metrics, forecast, residuals = run_forecast(model_type, target, tuple(features), horizon, method)
        except Exception as exc:  # noqa: BLE001 - surface model failures in the UI
            st.error(f"Model failed: {exc}")
            st.stop()

        mape_text = f"{metrics['mape']:.2f}%" if np.isfinite(metrics["mape"]) else "n/a"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² (backtest)", fmt(metrics["r2"]))
        c2.metric("RMSE", fmt(metrics["rmse"]))
        c3.metric("MAE", fmt(metrics["mae"]))
        c4.metric("MAPE", mape_text)

        fig = px.line(DATA, x="Year", y=target, title=f"{target}: history and forecast")
        fig.add_scatter(
            x=forecast["Year"],
            y=forecast["Predicted Price"],
            mode="lines+markers",
            name=f"Forecast ({horizon} yrs, {method})",
        )
        fig.update_layout(yaxis_title="Price (KES)")
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download forecast CSV",
            data=forecast.to_csv(index=False).encode("utf-8"),
            file_name=f"{target.replace(' ', '_').lower()}_forecast.csv",
            mime="text/csv",
        )

        if residuals is not None and not residuals.empty:
            st.caption("Prophet in-sample residuals")
            st.line_chart(residuals, x="Year", y="Residual")

    with st.expander("Data notes"):
        st.markdown(
            """
            - Rice prices are recorded as 0 for 2002-2020, so rice forecasts are unreliable.
            - The 2011 rainfall value (9 mm/year) looks like a data-entry error - verify it against the source.
            - Temperature varies only between 24-27 C, so it carries little signal.
            """
        )

    st.divider()
    if st.checkbox("Show raw data"):
        st.dataframe(DATA, use_container_width=True)


if __name__ == "__main__":
    main()
