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
    return render_template('index2.html')

# To use the predict button in our web-app


@app.route('/predict', methods=['POST'])
def predict():
    # For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    df = pd.DataFrame(final_features)
    df.head()
    features_name = ['age', 'sex', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'max_heart_rate_achieved', 'exercise_induced_angina',
                     'st_depression', 'num_major_vessels', 'chest_pain_type_asymptomatic', 'st_slope_type_flat', 'thalassemia_type_reversable_defect']

    df = pd.DataFrame(final_features, columns=features_name)
    prediction = model.predict(df[['exercise_induced_angina', 'st_depression', 'num_major_vessels',
                                   'chest_pain_type_asymptomatic', 'st_slope_type_flat', 'thalassemia_type_reversable_defect']])
    if prediction == 0:
        res_val = "Consult a Doctor"
    else:
        res_val = "You are healthy"

    return render_template('index2.html', prediction_text='{}'.format(res_val))


if __name__ == "__main__":
    app.run(debug=True)
