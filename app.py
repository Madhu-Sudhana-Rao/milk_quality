from flask import Flask, render_template, request, url_for, flash
import pickle
import os
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flash messages


# Load the ML model
def load_model():
    model_path = 'milk.pkl'
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as file:
        return pickle.load(file)


model = load_model()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        flash('Error: ML model not found. Please ensure milk.pkl is present in the application directory.', 'error')
        return render_template('index.html')

    try:
        # Get form data
        ph = float(request.form['pH'])
        temperature = int(request.form['Temprature'])
        taste = int(request.form['Taste'])
        odor = int(request.form['Odor'])
        fat = int(request.form['Fat'])
        turbidity = int(request.form['Turbidity'])
        colour = int(request.form['Colour'])

        # Prepare features for prediction (as a DataFrame with correct column names)
        features = pd.DataFrame([[ph, temperature, taste, odor, fat, turbidity, colour]],
                                columns=['pH', 'Temprature', 'Taste', 'Odor', 'Fat ', 'Turbidity', 'Colour'])

        # Make prediction using the loaded model
        prediction = model.predict(features)[0]

        # Convert numeric prediction to grade if necessary
        grade = str(prediction)

        return render_template('output.html', grade=grade)
    except Exception as e:
        flash(f'Error during prediction: {str(e)}', 'error')
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
