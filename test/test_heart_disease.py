import pandas as pd
import pytest

from src.heart_disease import (
    load_data,
    split_age_groups,
    count_chest_pain_types,
    train_model,
    prepare_model_data,
)


def test_load_data():
    df = load_data("data/cleanned-selected-columns.csv")

    assert not df.empty
    assert "age" in df.columns
    assert "thalch" in df.columns


def test_split_age_groups():
    test_df = pd.DataFrame({"age": [53, 54, 55]})

    younger, older = split_age_groups(test_df)

    assert len(younger) == 1
    assert len(older) == 2

    assert younger["age"].max() < 54
    assert older["age"].min() >= 54


def test_count_chest_pain_types():
    test_df = pd.DataFrame({"cp": [0, 0, 1, 2, 2, 2]})

    result = count_chest_pain_types(test_df)

    assert result[0] == 2
    assert result[1] == 1
    assert result[2] == 3


def test_train_model():
    df = load_data("data/cleanned-selected-columns.csv")

    model, predictions, y_test, mse = train_model(df)

    assert model is not None
    assert len(predictions) == len(y_test)
    assert mse >= 0


def test_prepare_model_data():
    df = pd.DataFrame(
        {
            "age": [45, 60, 50, 55, 65],
            "trestbps": [120, 140, 130, 135, 145],
            "chol": [200, 240, 210, 230, 250],
            "oldpeak": [0.0, 1.5, 0.5, 1.0, 2.0],
            "thalch": [170, 130, 160, 150, 120],
            "sex": [0, 1, 0, 1, 1],
        }
    )

    X, y = prepare_model_data(df)

    expected_X = df[["age", "trestbps", "chol", "oldpeak"]]
    expected_y = df["thalch"]

    pd.testing.assert_frame_equal(X, expected_X)
    pd.testing.assert_series_equal(y, expected_y)


def test_prepare_model_data_missing_column():
    df = pd.DataFrame(
        {
            "age": [45],
            "trestbps": [120],
            "oldpeak": [0.0],
            "thalch": [170],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns: chol"):
        prepare_model_data(df)


def test_prepare_model_data_excludes_zeros_without_changing_input():
    df = pd.DataFrame(
        {
            "age": [45, 60, 50, 55, 65, 40],
            "trestbps": [120, 140, 130, 135, 145, 125],
            "chol": [200, 0, 210, 230, 250, 190],
            "oldpeak": [0.0, 1.5, 0.5, 1.0, 2.0, 0.2],
            "thalch": [170, 130, 160, 150, 120, 175],
        },
        index=[10, 20, 30, 40, 50, 60],
    )
    original = df.copy(deep=True)

    X, y = prepare_model_data(df)

    expected_rows = df.loc[[10, 30, 40, 50, 60]]
    pd.testing.assert_frame_equal(
        X, expected_rows[["age", "trestbps", "chol", "oldpeak"]]
    )
    pd.testing.assert_series_equal(y, expected_rows["thalch"])
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize(
    "chol_values",
    [
        [0, 0, 0, 0, 0],
        [200, 210, 220, 230, 0],
    ],
    ids=["all-zero", "only-four-valid"],
)
def test_prepare_model_data_insufficient_records(chol_values):
    df = pd.DataFrame(
        {
            "age": [45, 60, 50, 55, 65],
            "trestbps": [120, 140, 130, 135, 145],
            "chol": chol_values,
            "oldpeak": [0.0, 1.5, 0.5, 1.0, 2.0],
            "thalch": [170, 130, 160, 150, 120],
        }
    )

    with pytest.raises(
        ValueError,
        match="At least 5 records with nonzero cholesterol are required",
    ):
        prepare_model_data(df)
