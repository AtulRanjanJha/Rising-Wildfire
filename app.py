from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')

# Load the trained model and scaler
with open("forest_fire_model.pkl", "rb") as model_file:
    model, scaler = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')  # Flask looks inside 'templates/'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]

    return jsonify({'prediction': 'fire' if prediction == 1 else 'not fire'})

if __name__ == '__main__':
    app.run(debug=True)
