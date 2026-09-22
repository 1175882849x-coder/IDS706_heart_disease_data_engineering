from src.heart_disease import (
    load_data,
    split_age_groups,
    count_chest_pain_types,
    train_model,
)

import pandas as pd


def test_full_workflow():
    # Load data
    df = load_data("data/cleanned-selected-columns.csv")

    # Split patients into age groups
    younger, older = split_age_groups(df)

    # Count chest pain types
    younger_cp = count_chest_pain_types(younger)
    older_cp = count_chest_pain_types(older)

    # Train and evaluate model
    model, predictions, y_test, mse = train_model(df)

    # Check the whole workflow
    assert not df.empty
    assert len(younger) + len(older) == len(df)

    assert younger_cp.sum() == len(younger)
    assert older_cp.sum() == len(older)

    assert model is not None
    assert len(predictions) == len(y_test)
    assert mse >= 0


# besides normal end to end workflow, test edge case for the workflow for multi components


def test_age_boundary_workflow():
    # Create a small dataset around the age cutoff of 54
    test_df = pd.DataFrame({"age": [53, 54, 55], "cp": [0, 1, 1]})

    # Run multiple components together
    younger, older = split_age_groups(test_df)

    younger_cp = count_chest_pain_types(younger)
    older_cp = count_chest_pain_types(older)

    # Verify the age groups are split correctly
    assert len(younger) == 1
    assert len(older) == 2

    # Verify chest pain counts remain consistent
    assert younger_cp.sum() == 1
    assert older_cp.sum() == 2

    # Age 54 is the boundary and should be in the older group
    assert 54 in older["age"].values
