import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler



application = Flask(__name__)

app=application

# Import Ridge regressor model and Standard scaler pickle 
ridge_model=pickle.load(open('D:\Yash\projects\Diabetes Prediction\Model\Modelforprediction.pkl','rb'))
standard_scaler=pickle.load(open('Model\standardscaler.pkl ','rb'))

## Route for home page

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    
    if request.method=='POST':
        
        Pregnancies=int(request.form.get("Pregnancies"))
        Glucose=float(request.form.get('Glucose'))
        BloodPressure=float(request.form.get("BloodPressure"))
        SkinThickness=float(request.form.get("SkinThickness"))
        Insulin=float(request.form.get("Insulin"))
        BMI=float(request.form.get("BMI"))
        DiabetesPedigreeFunction=float(request.form.get("DiabetesPedigreeFunction"))
        Age=float(request.form.get("Age"))
        

        new_data_scaled=standard_scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict=ridge_model.predict(new_data_scaled)
        
        if predict[0]==1:
            result="Diabetic"
        else:
            result='Non-Diabetic'
            
        
        return render_template('Single_predict.html',result=result)
    else:
        return render_template('home.html')

if __name__=="__main__":
    
    app.run(host = '0.0.0.0')
    
