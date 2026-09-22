# IDS706_heart-disease-data-engineering

[![Python Tests](https://github.com/1175882849x-coder/IDS706_heart_disease_data_engineering/actions/workflows/tests.yml/badge.svg)](https://github.com/1175882849x-coder/IDS706_heart_disease_data_engineering/actions/workflows/tests.yml)

## Project Overview

This project is part of the IDS706 Data Engineering course. The project explores a heart disease dataset using Pandas and Polars, performs basic data filtering and grouping, creates visualizations, and uses a simple machine learning model.

The project has also been extended to improve reproducibility and reliability by organizing reusable functions, adding automated unit and integration tests, and using GitHub Actions for continuous integration.

## Project Goal

The main goals of this project are to:

- Explore and summarize the heart disease dataset
- Compare patients across age groups
- Analyze chest pain type and cholesterol patterns
- Build a simple machine learning model to predict maximum heart rate
- Compare basic Pandas and Polars performance
- Create reusable data analysis functions
- Validate the workflow with automated testing
- Automatically run tests using GitHub Actions

## Data Source

Dataset: Heart Disease UCI Dataset  
https://www.kaggle.com/datasets/navjotkaushal/heart-disease-uci-dataset

The project uses a cleaned version of the dataset with **918 rows and 10 selected variables**.

## Project Structure

```text
IDS706_heart_disease_data_engineering/
├── .github/
│   └── workflows/
│       └── tests.yml
├── data/
│   └── cleanned-selected-columns.csv
├── notebook/
│   └── previous analysis notebooks
├── src/
│   ├── __init__.py
│   └── heart_disease.py
├── test/
│   ├── test_heart_disease.py
│   └── test_integration.py
├── week_3_assignment.ipynb
├── pytest.ini
├── requirements.txt
├── README.md
└── .gitignore
```

The reusable functions for the analysis are stored in `src/heart_disease.py`, while the automated tests are organized in the `test/` directory.

## Requirements

This project requires Python 3 and the following main libraries:

- pandas
- numpy
- matplotlib
- scikit-learn
- polars
- pytest
- jupyter

## Setup

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Analysis

The Week 3 analysis can be found in:

```text
week_3_assignment.ipynb
```

The notebook imports reusable functions from:

```text
src/heart_disease.py
```

This separates the core analysis logic from the notebook and makes the workflow easier to reuse and test.

## Data Analysis

The dataset was initially inspected using:

```python
df.head()
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
```

The dataset contains no missing values or duplicate rows.

The median age is **54**, so patients were divided into two groups:

```python
older_patients = df[df["age"] >= 54]
younger_patients = df[df["age"] < 54]
```

The two groups were compared using chest pain type (`cp`) and average cholesterol (`chol`).

### Average Cholesterol

- Younger patients: approximately **207.00**
- Older patients: approximately **193.84**

## Machine Learning

A Linear Regression model was used to predict maximum heart rate (`thalch`).

The input features were:

- Age (`age`)
- Resting blood pressure (`trestbps`)
- Cholesterol (`chol`)
- ST depression (`oldpeak`)

The feature and target variables were defined as:

```python
X = df[["age", "trestbps", "chol", "oldpeak"]]
y = df["thalch"]
```

The data was divided into:

- 80% training data
- 20% testing data

The model produced a Mean Squared Error of approximately:

```text
566.36
```

This model was used as an initial machine learning experiment rather than as a production prediction system.

## Visualization

The project includes:

- A boxplot comparing cholesterol distributions between younger and older patients
- A bar chart comparing average cholesterol between the two age groups
- A grouped bar chart comparing chest pain types between younger and older patients

These visualizations make differences between the two age groups easier to interpret.

## Testing

This project uses `pytest` for automated testing.

The current test suite contains **6 tests**:

### Unit Tests

The unit tests validate major components of the workflow, including:

- Data loading
- Age-group transformation
- Chest pain type grouping
- Machine learning model training and prediction

### Integration / System Tests

Integration tests validate interactions between multiple components and the end-to-end workflow.

They include:

- A full workflow test covering data loading, age grouping, chest pain analysis, model training, prediction, and evaluation
- An age-boundary test that checks the cutoff value of age 54 as an edge case

Run all tests locally with:

```bash
pytest -v
```

A successful run should report:

```text
6 passed
```

## Continuous Integration

GitHub Actions is used to automatically run the test suite whenever code is pushed to the `main` branch or a pull request is opened against `main`.

The workflow:

1. Checks out the repository
2. Sets up Python
3. Installs dependencies from `requirements.txt`
4. Runs the complete pytest test suite

The workflow configuration is stored in:

```text
.github/workflows/tests.yml
```

This helps ensure that changes to the project do not break existing functionality.

## Using Polars

Pandas and Polars were compared by reading the same CSV file and performing similar basic data operations.

For this dataset:

- Polars loaded the CSV in approximately **0.1 seconds**
- Pandas loaded the CSV in approximately **0.3 seconds**

Polars was faster for file reading in this experiment.

However, the dataset contains only 918 rows, so the overall performance difference is relatively small. Pandas and Polars also display results differently, with Polars using a more structured table-style format.

## Outcomes

The main outcomes of the project are:

- The dataset contains **918 observations and 10 selected variables**
- There are no missing values or duplicate rows
- The median age is **54**
- Younger patients had slightly higher average cholesterol in this dataset
- Older patients had more asymptomatic chest pain cases
- A Linear Regression model was successfully used as an initial machine learning experiment
- Core analysis logic was reorganized into reusable Python functions
- Four unit tests validate the major functions
- Two integration tests validate the workflow and an important age-boundary edge case
- All six tests pass successfully
- GitHub Actions automatically runs the tests after code changes

## Reproducibility

The project improves reproducibility by providing:

- A `requirements.txt` file for dependency installation
- Reusable source code in `src/`
- Automated tests using pytest
- Integration testing for the full workflow
- GitHub Actions continuous integration
- Clear instructions for running the analysis and tests

## Screenshots
<img width="1280" height="504" alt="bcb7ee4689056e1b1dde81619498a8a0" src="https://github.com/user-attachments/assets/83ac5cf1-0f38-434e-a406-7c7d105fe783" />
<img width="1007" height="224" alt="image" src="https://github.com/user-attachments/assets/b9f88bb7-9eee-4b44-8795-66ea61f3430b" />

