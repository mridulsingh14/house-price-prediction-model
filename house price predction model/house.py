import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Step 1: Load the dataset
df = pd.read_csv("house_prices.csv")  # Replace with your file path
print(df.head())  # Display first 5 rows

# Step 2: Exploratory Data Analysis (EDA)
print(df.info())  # Check for missing values
print(df.describe())  # Summary statistics

# Visualizing the data
plt.figure(figsize=(8,5))
sns.histplot(df["Price"], bins=30, kde=True)
plt.title("House Price Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 3: Handle Missing Values (if any)
df.fillna(df.mean(), inplace=True)

# Step 4: Split Data into Features (X) and Target (y)
X = df.drop("Price", axis=1)  # Features
y = df["Price"]  # Target variable

# Step 5: Split into Train & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Evaluate Model Performance
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R² Score: {r2}")

# Step 8: Predict House Price for New Data (Ensure Consistent Columns)
# Make sure the columns match those used during training
new_house = pd.DataFrame([[3200, 4, 10]], columns=["Area (sq ft)", "Bedrooms", "Age"])

# Make prediction
predicted_price = model.predict(new_house)
print(f"Predicted House Price: ${predicted_price[0]:,.2f}")

# Step 9: Save Model for Future Use
joblib.dump(model, "house_price_model.pkl")
print("Model saved successfully!")

# Step 10: Load & Test the Saved Model
loaded_model = joblib.load("house_price_model.pkl")

# Test prediction with new data
test_house = pd.DataFrame([[2800, 3, 5]], columns=["Area (sq ft)", "Bedrooms", "Age"])
test_price = loaded_model.predict(test_house)
print(f"Predicted House Price for Test Data: ${test_price[0]:,.2f}")
