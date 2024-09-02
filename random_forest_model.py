import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset
df = pd.read_csv("melb_data.csv")

# Preprocessing
df = df.dropna(subset=['Price'])  # Drop rows where 'Price' is missing
df.fillna({
    'Bedroom2': df['Bedroom2'].median(),
    'Bathroom': df['Bathroom'].median(),
    'Car': df['Car'].median(),
    'Landsize': df['Landsize'].median(),
    'BuildingArea': df['BuildingArea'].median(),
    'YearBuilt': df['YearBuilt'].median(),
}, inplace=True)

# Encode categorical features
df = pd.get_dummies(df, columns=['Suburb', 'Type', 'Method', 'SellerG', 'CouncilArea', 'Regionname'])

# Drop irrelevant columns
df.drop(['Address', 'Date'], axis=1, inplace=True)

# Split data into features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Save the model and the scaler
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and Scaler saved!")
