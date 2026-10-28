#1 - import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#2 - load datasets
orders2015 = pd.read_csv("../Dataset/order_2015.csv")
orders2016 = pd.read_csv("../Dataset/order_2016.csv")
orders2017 = pd.read_csv("../Dataset/order_2017.csv")
orders2018 = pd.read_csv("../Dataset/order_2018.csv")

people = pd.read_csv("../Dataset/People.csv")
returns = pd.read_csv("../Dataset/Returns.csv")
target = pd.read_csv("../Dataset/Target.csv")

#3 - merge all order datasets
orders = pd.concat(
    [orders2015, orders2016, orders2017, orders2018],
    ignore_index = True
)

print("Dataset Loaded Successfully")
print()

print("Shape : ", orders.shape)
print()

print("Columns")
print(orders.columns)
print()

print(orders.head())

#4 - data cleaning
#missing values
print("\nmissing values")
print(orders.isnull().sum())

#duplicate records
print("\nduplicate rows : ", orders.duplicated().sum())

#remove duplicates
orders = orders.drop_duplicates()
print("Shape after removing duplicates : ", orders.shape)

#convert excel dates
orders["Order Date"] = pd.to_datetime(
    orders["Order Date"],
    unit = "D",
    origin = "1899-12-30" 
)

orders["Ship Date"] = pd.to_datetime(
    orders["Ship Date"],
    unit = "D",
    origin = "1899-12-30"
)

print("\nConverted Dates")
print(orders[["Order Date", "Ship Date"]].head())

#5 - feature engineering
orders["Year"] = orders["Order Date"].dt.year
orders["Month"] = orders["Order Date"].dt.month
orders["Quarter"] = orders["Order Date"].dt.quarter
print()
print(
    orders[["Order Date", "Year", "Month", "Quarter"]].head()
)

#Merging....
print("Merged dataset of people & region based on orders")
#6 - merge people dataset
print()
orders = pd.merge(
    orders,
    people,
    on = "Region",
    how = "left"
)
print("\nOrders after merging people")
print()
print(orders.head())

print("\nShape : ", orders.shape)

#7 - merge returns dataset 
#7.1 - checking for duplicate order ID in returns
print("Duplicate Order ID in REturns")
print()
print(returns["Order ID"].duplicated().sum())

#7.2 - remove duplicate order id from returns
returns = returns.drop_duplicates(subset = "Order ID")

print("Returns Shape : ",returns.shape)

#7.3 - Merge returns
orders = pd.merge(
    orders,
    returns,
    on = "Order ID",
    how = "left"
)

orders["Returned"] = orders["Returned"].fillna("No")
print("\nOrders after merging Returns")
print()
print(
    orders[
        ["Order ID", "Returned"]
    ].head()
)
print()
print("Shape : ",orders.shape)

#save checkpoint 1
print()
orders.to_csv("../Output/orders_after_returns.csv",index=False)
print("Checkpoint saved sucessfully...")
print()

#8 - merger target 
#8.1 - clean target dataset
target["Sales Target"] = (
    target["Sales Target"].str.replace("K","",regex=False)
    .astype(int)*1000
)
print(target.head())
print()

#8.2 - merge target dataset
orders = pd.merge(
    orders,
    target,
    on=["Category","Year"],
    how="left"
)
print("after target merge")
print(orders[
    ["Category", "Year", "Sales Target"]].head())
print()

#cleaning quantity column

quantity_mapping = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10
}
orders["Quantity"] = orders["Quantity"].replace(quantity_mapping)

orders["Quantity"] = pd.to_numeric(
    orders["Quantity"],
    errors="coerce"
)

orders["Quantity"] = orders["Quantity"].fillna(0).astype(int)

print("\nQuantity cleaned successfully")
print(orders["Quantity"].dtype)
print(orders["Quantity"].unique())
print()
print("\nShape : ",orders.shape)

#9 - Final dataset validation
print("\n Final Dataset Validation")

print("\nShape : ")
print(orders.shape)

print("\nMissing Values : ")
print(orders.isnull().sum())

print("\nDuplicate Rows : ")
print(orders.duplicated().sum())

print("\nData Types : ")
print(orders.dtypes)

print("\nSummary Statistics : ")
print(orders.describe())

#saving final dataset

orders.to_csv(
    "../Output/final_validated_dataset.csv",
    index=False
)

print("\nFinal Validated dataset saved successfully")
print()

#inspect additional dataset
#print("\nPeople Dataset")
#print(people.head())
#print(people.columns)

#print("\nReturns Dataset")
#print(returns.head())
#print(returns.columns)

#print("\nTarget Dataset")
#print(target.head())
#print(target.columns)

print()
#10 - Exploratory Data Analysis(EDA)
#step 10.1
print("EDA")

#Basic KPIs - key Performance Indicators
#step 10.2
total_sales = orders["Sales"].sum()
total_profit = orders["Profit"].sum()
total_orders = orders["Order ID"].nunique()

print("\nBasic KPIs")
print(f"Total Sales: {total_sales:.2f}")
print(f"Total Profit: {total_profit:.2f}")
print(f"Total Orders: {total_orders}")

#sales by region
#step 10.3
sales_region = (
    orders.groupby("Region")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSales by Region")
print(sales_region)

#profit by Region
#step 10.4
profit_region = (
    orders.groupby("Region")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

print("\nProfit by Region")
print(profit_region)

#sales by category
#step 10.5
sales_category = (
    orders.groupby("Category")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSales by Category")
print(sales_category)

#profit by category
#step 10.6
profit_category = (
    orders.groupby("Category")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

print("\nProfit by CAtegory")
print(profit_category)

print()

#11 - Graph
#11.1 - sales by region chart

plt.figure(figsize = (8,5))

sales_region.plot(kind = "bar")

plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")

plt.tight_layout()

plt.savefig("../Graphs/sales_by_region.png")
plt.show()

#11.2 - profit by category

plt.figure(figsize = (8,5))

profit_category.plot(kind = "bar")

plt.title("Profit by Category")
plt.xlabel("CAtegory")
plt.ylabel("Profit")

plt.tight_layout()

plt.savefig("../Graphs/prfit_by_category.png")
plt.show()

#11.3 Sales by year
sales_year = orders.groupby("Year")["Sales"].sum()

plt.figure(figsize=(8,5))

sales_year.plot(kind="line", marker = "o")
plt.title("Sales by Year")
plt.xlabel("Year")
plt.ylabel("Sales")

plt.grid(True)
plt.tight_layout()
plt.savefig("../Graphs/sales_by_year.png")
plt.show()

#11.4 sales by month
sales_month = orders.groupby("Month")["Sales"].sum()
plt.figure(figsize=(8,5))

sales_month.plot(kind="line", marker="o")

plt.title("Monthly Sales Tremd")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.grid(True)
plt.tight_layout()
plt.savefig("../Graphs/monthly_sales.png")
plt.show()

#11.5 Return analysis
return_count = orders["Returned"].value_counts()
plt.figure(figsize=(6,6))

return_count.plot(kind="pie", autopct = "%1.1f%%")

plt.ylabel("")
plt.title("Returned Orders")
plt.savefig("../Graphs/returns.png")
plt.show()

#11.6 Correlation analysis
correlation = orders[
    ["Sales","Profit","Quantity","Discount","Sales Target"]
    ].corr()

print("\nCorrelation Matrix")
print(correlation)
print()

#12 - FEature Selection
print("Feature Selection")

#12.1 select the features
#input features

x = orders[
    [
        "Category",
        "Sub-Category",
        "Region",
        "Segment",
        "Quantity",
        "Discount",
        "Sales Target"
    ]
] #feature matrix

y = orders["Sales"] #target variable

print("\nInput Features")
print(x.head())

print("\nTarget Variable")
print(y.head())
print()

#13 Data Encoding (One-Hot Encoding)
print("DAta Encoding")

x_encoded = pd.get_dummies(
    x,
    columns=[
        "Category",
        "Sub-Category",
        "Region",
        "Segment"
    ],
    drop_first=True
)

print("\nEncoded Features")
print(x_encoded.head())
print("\nShape : ", x_encoded.shape)
print()

#14 - Train Test Split
print("Train Test Split")

x_train, x_test, y_train, y_test = train_test_split(
    x_encoded,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape")
print(x_train.shape)

print("\nTesting Data Shape")
print(x_test.shape)
print()

#15 Linear Regression Model
#training model
print("Model Training")

model = LinearRegression()
model.fit(x_train, y_train)

print("\nModel Trained successfully")
print()

#16 sales prediction
print("sales prediction")

predictions = model.predict(x_test)
print("\n first 10 prediction")

for i in range(10):
    print(
        f"Actual : {y_test.iloc[i]:.2f} | Predicted : {predictions[i]:.2f}"
    )

#17 Model Evaluation (outcome)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nModel Performance Matrics")
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2 Score : {r2:.4f}")

#18 Actual vs prsdicted plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color = "red", linestyle = "--"
)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.tight_layout()
plt.savefig("../Graphs/actual_vs_predicted.png")
plt.show()

#Feature Importance (coefficients)
coeff_df = pd.DataFrame({
    "Feature": x_encoded.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print()
print("\nFeature Importance (Model Coefficients)")
print(coeff_df)