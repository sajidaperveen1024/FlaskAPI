
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

app = Flask(__name__)

# Route for training the model
@app.route('/train', methods=['POST'])
def train():
    # Load the dataset
    df = pd.read_csv("data/melb_data.csv")

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

    return jsonify({"message": "Model and Scaler saved successfully!"})


# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Load the model and scaler
    if not os.path.exists('random_forest_model.pkl') or not os.path.exists('scaler.pkl'):
        return jsonify({"error": "Model or Scaler not found, please train the model first!"}), 400

    rf_model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Get the JSON data from the request
    data = request.get_json(force=True)

    # Convert the JSON data to a DataFrame
    df = pd.DataFrame(data)

    # Encode categorical features (assumes you receive the same categorical columns as in training)
    df = pd.get_dummies(df)

    # Reindex to match training data columns
    X = df.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # Scale the features
    X_scaled = scaler.transform(X)

    # Make predictions
    predictions = rf_model.predict(X_scaled)

    # Return the predictions as a JSON response
    return jsonify({"predictions": predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)




# from flask import Flask, request, jsonify
# import joblib
# import numpy as np

# app = Flask(__name__)

# # Load the Random Forest model and the scaler
# rf_model = joblib.load('random_forest_model.pkl')
# scaler = joblib.load('scaler.pkl')

# @app.route('/')
# def home():
#     return "Welcome to the House Price Prediction API!"

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     try:
#         # Extract features from the request data
#         features = np.array([
#             data['location'],  # Assuming you map the location to a numerical value
#             data['bhk'],
#             data['bath'],
#             data['total_sqft']
#         ]).reshape(1, -1)

#         # Scale the features
#         scaled_features = scaler.transform(features)

#         # Make prediction
#         prediction = rf_model.predict(scaled_features)

#         # Return prediction as JSON
#         return jsonify({'price': prediction[0]})

#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)
