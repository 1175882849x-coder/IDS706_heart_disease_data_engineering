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
            "age": [45, 60],
            "trestbps": [120, 140],
            "chol": [200, 240],
            "oldpeak": [0.0, 1.5],
            "thalch": [170, 130],
            "sex": [0, 1],
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
