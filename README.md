# AI-Driven Sales Analytics and Sales Prediction Using Machine Learning

## Project Overview

This project analyzes historical sales data to identify sales trends, evaluate business performance, and build a machine learning model for sales prediction.

It combines data preprocessing, exploratory data analysis, visualization, feature analysis, and predictive modeling using Python and Scikit-learn.

## Objectives

- Analyze historical sales performance across different years and business categories.
- Visualize sales trends, regional performance, and profitability.
- Investigate the relationship between input features and sales outcomes.
- Build and evaluate a machine learning model for sales prediction.
- Compare model performance using standard regression evaluation metrics.

## Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib
- **Machine Learning:** Scikit-learn
- **Version Control:** Git and GitHub

## Project Structure

```text
AI-Driven-Sales-Analytics/
├── Dataset/          # Input datasets (share only if authorized)
├── Graphs/           # Generated visualizations
├── Output/           # Analysis results and evaluation tables
├── Presentation/     # Project presentation material
├── Python/           # Python implementation
├── .gitignore
├── requirements.txt
└── README.md
```

## Methodology

1. **Data Preparation:** Load and preprocess the historical sales data.
2. **Exploratory Data Analysis:** Examine sales trends, regional performance, categories, and returns.
3. **Data Visualization:** Generate graphs to communicate key findings.
4. **Feature Analysis:** Examine feature importance and evaluate the contribution of selected features.
5. **Model Development:** Implement a Linear Regression model using Scikit-learn.
6. **Model Evaluation:** Assess predictions using R², Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

## Model Evaluation Results

The reported test-set evaluation results are:

| Metric | Result |
|---|---:|
| R² Score | 0.4072 |
| Mean Absolute Error (MAE) | 198.46 |
| Root Mean Squared Error (RMSE) | 331.70 |

The R² score indicates that the model explains approximately 40.72% of the variance in the target variable on the evaluated test set. The MAE and RMSE quantify prediction errors in the target variable's units.

These results provide a baseline for further feature engineering and model improvement.

## Visualizations

The project includes visualizations such as:

- Actual vs. predicted sales
- Monthly and yearly sales trends
- Sales performance by region
- Category-level profitability
- Feature importance
- Sales quartile analysis
- Returns analysis

See the `Graphs/` directory for the generated figures.

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/Kiran25005369-hub/AI-Driven-Sales-Analytics.git
cd AI-Driven-Sales-Analytics
```

### 2. Create a virtual environment (recommended)

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Run the project

Run the main Python script:

```powershell
.\.venv\Scripts\python.exe Python/main.py
```

Check the `Python/` directory for the available scripts and their specific functionality.

## Future Improvements

- Evaluate additional regression algorithms.
- Improve feature engineering and hyperparameter tuning.
- Investigate prediction errors across different sales segments.
- Compare random test splits with time-based validation.
- Develop an interactive dashboard for sales analysis.

## Author

**Kiran**

M.Sc. Computer Science and Information Technology

---