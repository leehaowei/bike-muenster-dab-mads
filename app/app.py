from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from joblib import load
app = Flask(__name__)

cols = ['temperature', 'humidity', 'rain_mm', 'rain_day', 'h_0', 'h_1', 'h_2',
       'h_3', 'h_4', 'h_5', 'h_6', 'h_7', 'h_8', 'h_9', 'h_10', 'h_11', 'h_12',
       'h_13', 'h_14', 'h_15', 'h_16', 'h_17', 'h_18', 'h_19', 'h_20', 'h_21',
       'h_22', 'h_23', 'm_1', 'm_2', 'm_3', 'm_4', 'm_5', 'm_6', 'm_7', 'm_8',
       'm_9', 'm_10', 'm_11', 'm_12']

# test_lst = [11, 31, 8, 9, 82, 0, 0]

@app.route('/', methods=['GET', 'POST'])
def predict_bike():
    # import model and scaler
    rf_model = load('app/model.joblib')
    sc = load('app/std_scaler.joblib')

    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html', href="static/x.svg")
    else:
        month = request.form['month']
        day = request.form['day']
        hour = request.form['hour']
        temp = request.form['temp']
        humd = request.form['humd']
        rmm = request.form['rmm']
        rain = request.form['rain']
        list_ = [month, hour, temp, humd, rmm, rain]
        prediction = model_predict(list_, rf_model, sc)
        str_ = f'The predicted bike traffic at {hour}:00 on {day}/{month} is {prediction} bikes'
        return str_


def model_predict(in_, model, sc):
    arr = np.array(in_)

     # preprocessing
    empty_df = pd.DataFrame(np.array([0 for i in range(40)]).reshape(1,40), columns=cols)
    month = arr[0]
    hour = arr[1]
    col_month = 'm_' + str(month)
    col_hour = 'h_' + str(hour) 

    # fill in the data
    empty_df[col_month] = 1
    empty_df[col_hour] = 1
    empty_df['temperature'] = arr[2]
    empty_df['humidity'] = arr[3]
    empty_df['rain_mm'] = arr[4]
    empty_df['rain_day'] = arr[5]

    arr = empty_df.values
    # standardisation
    standardised_data = sc.transform(arr)  # always remember to standadise

    # predict by random forest
    rf_pred = model.predict(standardised_data)
    prediction = int(rf_pred[0])
    return str(prediction)