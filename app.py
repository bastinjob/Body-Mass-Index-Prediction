import numpy as np
import pandas as pd
import pickle
#from preprocess import preprocess_data

from flask import Flask,request, jsonify, render_template


#initialize the app

app = Flask(__name__)


with open('models/random_forest_classifier_model.pkl', 'rb') as f:
    model = pickle.load(f)



dict_bmi = {
            0: 'Extremely Weak',
            1: 'Weak',
            2: 'Normal',
            3: 'Overweight',
            4: 'Obese',
            5: 'Extremely Obese'
        }

def preprocess_inputs(gender, height, weight):

    
    gender_encoded  = 1. if gender.lower() == 'male' else 0.

    input_df = pd.DataFrame({'Gender_encoded':[gender_encoded], 'Height':[height], 'Weight':[weight]})


    return input_df.values

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if 'file' in request.files:
        file = request.files['file']
        df = pd.read_csv(file)
        df['Gender_encoded'] = df['Gender'].apply(lambda gender: 1 if gender.lower() == 'male' else 0)
        scaled_features = scaler.transform(df[['Height', 'Weight']])
        df[['Height', 'Weight']] = scaled_features
        predictions = model.predict(df[['Gender_encoded', 'Height', 'Weight']])
        df['BMI Category'] = predictions
        df['Obesity Level'] = df['BMI Category'].apply(lambda pred: dict_bmi[pred])

        #return jsonify({'BMI Category':predictions.tolist(),'Obesity Level': Obesity_levels })

        result_html = df.to_html(classes='table table-striped')

        return render_template('result.html', table=result_html)

    else:
        data = request.form
        gender = data.get('gender')
        height = data.get('height')
        weight = data.get('weight')

        if gender is None or height is None or weight is None:
            return jsonify({'error': 'Missing input data'}), 400

        input_data = preprocess_inputs(gender, height, weight)
        prediction = model.predict(input_data)

        df = pd.DataFrame({'Gender':[gender], 'Height':[height], 'Weight':[weight]})
        df['BMI Category'] = [prediction[0]]
        df['Obesity Level'] = [dict_bmi[prediction[0]]]
        
        #obesity = dict_bmi[prediction]
        #return jsonify({'BMI': prediction})

        result_html = df.to_html(classes='table table-striped')

        return render_template('result.html', table=result_html)
    

if __name__ == '__main__':
    app.run(debug=True)


