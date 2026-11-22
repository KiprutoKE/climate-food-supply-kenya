"""Plotly figure builders for the dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .data import PCT_SUFFIX


def price_trend(df: pd.DataFrame, crops: list[str]) -> go.Figure:
    long = df.melt(id_vars="Year", value_vars=list(crops), var_name="Crop", value_name="Price (KES)")
    fig = px.line(long, x="Year", y="Price (KES)", color="Crop", markers=True, title="Crop prices over time")
    fig.update_layout(legend_title_text="Crop")
    return fig


def production_index(df: pd.DataFrame) -> go.Figure:
    return px.line(df, x="Year", y="Food Production Index", markers=True, title="Food Production Index over time")


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    corr = df.select_dtypes(include="number").corr()
    return px.imshow(
        corr,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        text_auto=".1f",
        title="Correlation matrix",
        labels={"color": "Correlation"},
    )


def scatter_pair(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    return px.scatter(df, x=x, y=y, trendline="ols", title=f"{x} vs {y}")


def price_change_distribution(df: pd.DataFrame, crops: list[str]) -> go.Figure:
    columns = [f"{c}{PCT_SUFFIX}" for c in crops]
    long = df.melt(value_vars=columns, var_name="Crop", value_name="Yearly % Change")
    long["Crop"] = long["Crop"].str.replace(PCT_SUFFIX, "", regex=False)
    return px.box(
        long,
        x="Crop",
        y="Yearly % Change",
        points="all",
        color="Crop",
        title="Distribution of year-over-year price changes",
    )
