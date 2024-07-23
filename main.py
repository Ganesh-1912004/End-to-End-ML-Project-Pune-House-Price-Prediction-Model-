from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load and preprocess the CSV data
data = pd.read_csv('cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))
@app.route('/')
def index():

    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bath'))
    sqft = request.form.get('total_sqft')

    print(location, bhk, bath, sqft)
    input = pd.DataFrame([[location, bhk, sqft, bath]], columns=['location', 'BHK', 'total_sqft', 'bath'])
    prediction = pipe.predict(input)[0] * 1e5

    return str(np.round(prediction,2))

if __name__ == '__main__':
    app.run(debug=True)
