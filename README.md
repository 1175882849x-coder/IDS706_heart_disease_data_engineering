# IDS706_heart-disease-data-engineering

## Project Goal

This project is part of the IDS706 Data Engineering course. The goal is to explore a heart disease dataset using Pandas, perform basic filtering and grouping, create visualizations, and experiment with a simple machine learning model.

## Data Source

Dataset: Heart Disease UCI Dataset  
https://www.kaggle.com/datasets/navjotkaushal/heart-disease-uci-dataset

The project uses a cleaned version of the dataset with 918 rows and 10 selected variables.

## Requirements

This project requires Python 3 and the following libraries:

- pandas
- matplotlib
- scikit-learn
- jupyter

## Set up

python -m pip install pandas matplotlib scikit-learn jupyter

## Data Analysis
The dataset was inspected using:
df.head()
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()

The median age is 54, so I divided the data into two subsets:
older_patients = df[df["age"] >= 54]
younger_patients = df[df["age"] < 54]

The two groups were compared using chest pain type (cp) and average cholesterol (chol).

Average cholesterol:

Younger patients: about 207.00
Older patients: about 193.84

## Machine Learning
I used Linear Regression to predict maximum heart rate (thalch).

Input features:

age
resting blood pressure (trestbps)
cholesterol (chol)
ST depression (oldpeak)

The data was split into 80% training data and 20% testing data.

X = df[["age", "trestbps", "chol", "oldpeak"]]
y = df["thalch"]

The model produced a Mean Squared Error of approximately: 566.36

## Visualization
I created:

a boxplot comparing cholesterol distributions between younger and older patients
a bar chart comparing average cholesterol between the two age groups
a grouped bar chart comparing chest pain types between younger and older patients

These visualizations help make the group differences easier to interpret.

## Outcomes
The dataset has 918 observations and 10 variables.
There are no missing values or duplicate rows.
The median age is 54.
Younger patients had slightly higher average cholesterol in this dataset.
Older patients had more asymptomatic chest pain cases.
A Linear Regression model was successfully used as an initial machine learning experiment.

## Using Polars

I compared Pandas and Polars by reading the same CSV file and performing similar basic data operations.

For this dataset, Polars loaded the CSV in about 0.1 seconds, while Pandas took about 0.3 seconds. This suggests that Polars was faster for file reading in this test.

However, the dataset is relatively small, with only 918 rows, so the overall performance difference is still minor. I also noticed that Pandas and Polars display their outputs differently, with Polars using a more structured table-style format.