from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model and handle potential errors
def load_model():
    try:
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    except FileNotFoundError:
        print("Model file not found. Ensure that 'model.pkl' is in the correct directory.")
        return None

# Load the model during initialization
model = load_model()

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if model:
        # Get input data from the form
        try:
            features = [float(x) for x in request.form.values()]
            final_features = np.array([features])
            
            # Make a prediction using the pre-trained model
            prediction = model.predict(final_features)
            
            # Format the output
            output = round(prediction[0], 2)
            return render_template('index.html', prediction_text=f'Predicted House Price: ${output}')
        
        except ValueError as e:
            return render_template('index.html', prediction_text=f'Error in input values: {str(e)}')
    else:
        return render_template('index.html', prediction_text="Model not loaded, cannot make predictions.")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
