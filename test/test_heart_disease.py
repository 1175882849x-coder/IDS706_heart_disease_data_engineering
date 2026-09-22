import pandas as pd

from src.heart_disease import (
    load_data,
    split_age_groups,
    count_chest_pain_types,
    train_model,
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
