import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def load_data(filepath):
    """
    Load the heart disease dataset from a CSV file.
    """
    df = pd.read_csv(filepath)
    return df


def split_age_groups(df, cutoff=54):
    """
    Split patients into younger and older groups based on age.
    """
    younger_patients = df[df["age"] < cutoff]
    older_patients = df[df["age"] >= cutoff]

    return younger_patients, older_patients


def count_chest_pain_types(df):
    """
    Count the number of patients for each chest pain type.
    """
    cp_counts = df.groupby("cp").size()
    return cp_counts


def train_model(df):
    """
    Train a linear regression model to predict maximum heart rate (thalch).
    """
    X = df[["age", "trestbps", "chol", "oldpeak"]]
    y = df["thalch"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    return model, predictions, y_test, mse