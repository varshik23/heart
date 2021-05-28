#import libraries
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle
# Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# default page of our web-app


@app.route('/')
def home():
    return render_template('index.html')

# To use the predict button in our web-app


@app.route('/predict', methods=['POST'])
def predict():
    # For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    df = pd.DataFrame(final_features)
    df.head()
    prediction = model.predict(df)
    if prediction == 0:
        return render_template('index.html', prediction_text='Person has heart disease')
    else:
        return render_template('index.html', prediction_text='You are healthy')


if __name__ == "__main__":
    app.run(debug=True)
