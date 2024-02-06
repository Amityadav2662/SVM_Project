from flask import Flask, render_template, app, request
from flask import Response
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import numpy as np

application = Flask(__name__)
app = application

# Taking the Model for futher execaution
svm = pickle.load(open("model/SVM.pkl","rb"))
scalar = pickle.load(open("model/Standard_Scaler.pkl","rb"))

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Pregnancies = int(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = int(request.form.get('Age'))

        new_data_scaled = scalar.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        prediction = svm.predict(new_data_scaled)

        if prediction[0] == 1:
            result = 'Diabetic'
        else:
            result = 'No Diabetic'

        return render_template('single_prediction.html',result=result)
    else:
        return render_template('home.html')



if __name__=="__main__":
    app.run(host="0.0.0.0")
