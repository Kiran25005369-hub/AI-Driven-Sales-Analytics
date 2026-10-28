# AI-Driven Sales Analytics and Sales Prediction Using Machine Learning

## Overview
This project analyzes historical retail sales data and develops a machine learning model to predict sales. It demonstrates an end-to-end workflow covering data preprocessing, exploratory data analysis, feature engineering, categorical encoding, model training, and evaluation.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- Linear Regression

## Project Workflow
1. Load and integrate the order datasets.
2. Clean and validate the data.
3. Engineer time-based features.
4. Perform exploratory data analysis and visualization.
5. Select model features and encode categorical variables.
6. Split the data into training and testing sets.
7. Train a Linear Regression model.
8. Evaluate predictions using MAE, RMSE, and R².

## Baseline Model Results
- **R²:** 0.4072
- **MAE:** 198.46
- **RMSE:** 331.70

These results represent the Linear Regression baseline from the current implementation. Further experiments are needed to determine whether other models improve performance.

## How to Run
1. Install Python.
2. Clone or download this repository.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Ensure the permitted input datasets are in the expected locations.
5. Run:

   `python main.py`

Update the dataset paths if necessary.

## Limitations and Future Improvements
- Compare Linear Regression with Random Forest and Gradient Boosting.
- Investigate additional features and validation strategies.
- Improve prediction accuracy and evaluate generalization.

## Author
Kiran A — M.Sc. Computer Science and Information Technology

## Data and Usage
The datasets are not included unless their redistribution is permitted. Users must obtain authorized data separately and configure the paths accordingly.
