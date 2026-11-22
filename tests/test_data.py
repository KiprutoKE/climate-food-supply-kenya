from src.data import PCT_SUFFIX, PRICE_COLUMNS, load_data


def test_load_data_shape_and_columns():
    df = load_data()
    assert len(df) == 24
    assert {"Year", "Food Production Index", *PRICE_COLUMNS} <= set(df.columns)


def test_pct_change_first_year_is_zero():
    df = load_data()
    for column in PRICE_COLUMNS:
        assert df[f"{column}{PCT_SUFFIX}"].iloc[0] == 0


def test_no_nans_in_numeric_columns():
    df = load_data()
    assert not df.select_dtypes("number").isna().any().any()
