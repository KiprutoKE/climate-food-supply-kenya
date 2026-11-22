# Influences of Climatic Patterns on Food Supply in Kenya

An interactive Streamlit dashboard that explores how temperature and rainfall relate to
staple crop prices in Kenya, and forecasts prices a few years ahead with three models:
Gaussian Process Regression, Random Forest and Prophet.

## Features

- **Exploratory analysis** - price trends, Food Production Index, correlation matrix,
  pairwise scatter plots with an OLS trend line, and the distribution of year-over-year
  price changes.
- **Model predictions** - pick a crop, feature columns, model, forecast horizon (1-10 years)
  and a future-driver scenario (`trend` or `constant`). Results include backtest metrics
  (R², RMSE, MAE, MAPE), a history + forecast chart and a CSV download.
- **Data-quality notes** surfaced directly in the app.

## Dataset

`data/data.csv` has 24 annual rows (2002-2025) with:

| Column | Meaning |
| --- | --- |
| `Year` | Calendar year |
| `Food Production Index` | National food production index |
| `Beans_Price_90KG`, `Maize_Price_90KG`, `Potatoes_Price_50KG`, `Rice_Price_50KG` | Crop prices (KES) |
| `Temperature - (Celsius)` | Annual mean temperature |
| `Rainfall (mm/year)` | Annual rainfall |

Known issues to verify against the original source before drawing conclusions:

- Rice price is `0` for 2002-2020, so rice forecasts are unreliable.
- The 2011 rainfall value is `9` mm/year, which looks like a data-entry error.
- Temperature only ranges from 24-27 C, so it carries little signal.

## Project structure

```
climate-food-supply-kenya/
├── streamlit_app.py        # Streamlit entry point (Streamlit Cloud default)
├── data/data.csv           # Dataset
├── src/
│   ├── charts.py           # Plotly figure builders
│   ├── data.py             # Loading, cleaning and feature engineering
│   └── models.py           # GPR / Random Forest / Prophet forecasting
├── tests/test_data.py      # Data-loader tests
├── .streamlit/config.toml  # Theme
├── requirements.txt        # Pinned runtime dependencies
└── requirements-dev.txt    # Adds pytest
```

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open https://climate-food-supply-chain-in-kenya.streamlit.app/

## Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> -> **Create app** -> **Deploy a public app from GitHub**.
2. Repository: `KiprutoKE/climate-food-supply-kenya`, branch: `main`.
3. **Main file path:** `streamlit_app.py` (the default the form pre-fills).
4. Optionally pick Python 3.11+ under **Advanced settings**, then click **Deploy**.

`requirements.txt` sits at the repository root, so Streamlit installs the pinned
dependencies automatically. The first build takes a few minutes because Prophet
pulls in `cmdstanpy`; the app still starts without Prophet if that install fails.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Methodology and caveats

- Cross-validation uses `TimeSeriesSplit` (not shuffled K-Fold) so future years are never
  used to train on the past.
- MAPE is computed only over non-zero actuals, because several crop prices are `0`.
- Future drivers are projected with either a least-squares trend over the full history or a
  constant (last-value) scenario. These are scenario projections, not real climate forecasts.
- Features include the same-year Food Production Index and other crops' price changes. That
  makes the model a **scenario tool** rather than a true out-of-sample price forecaster; use
  lagged features if you need a strict forecasting setup.
- Random Forest cannot extrapolate beyond the training range, so its future predictions
  tend to flatten out.

## License

MIT - see [LICENSE](LICENSE).
