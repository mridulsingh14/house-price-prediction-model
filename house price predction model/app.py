import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("house_price_model.pkl")
print("✅ Model loaded successfully!")


# Function to predict the house price based on user input
def predict_house_price():
    """
    Takes user input for area, bedrooms, and age of the house, then predicts the price.
    """
    # Taking user input for house features
    area = float(input("Enter house area (sq ft): "))
    bedrooms = int(input("Enter number of bedrooms: "))
    age = int(input("Enter house age (years): "))

    # Create a Pandas DataFrame to match the training format
    input_data = pd.DataFrame({
        'Area (sq ft)': [area],  # Area in sq ft (corrected column name)
        'Bedrooms': [bedrooms],  # Number of bedrooms
        'Age': [age]  # Age of the house
    })

    # Predict the house price
    predicted_price = model.predict(input_data)

    # Return the predicted price
    print(f"💰 Predicted House Price: ${predicted_price[0]:,.2f}")


# Call the function to make predictions
predict_house_price()
