#1 - import libraries
import pandas as pd

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