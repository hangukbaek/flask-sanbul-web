import tensorflow as tf
from tensorflow import keras
import joblib

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
full_pipeline = joblib.load("pipeline.pkl")

import numpy as np
import pandas as pd
from flask import Flask, render_template

from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

STRING_FIELD = StringField('max_wind_speed', validators=[DataRequired()])

np.random.seed(42)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap5 = Bootstrap5(app)

class LabForm(FlaskForm):
    longitude = StringField('longitude(1-7)', validators=[DataRequired()])
    latitude = StringField('latitude(1-7)', validators=[DataRequired()])
    month = StringField('month(01-Jan ~ Dec-12)', validators=[DataRequired()])
    day = StringField('day(00-sun ~ 06-sat, 07-hol)', validators=[DataRequired()])
    avg_temp = StringField('avg_temp', validators=[DataRequired()])
    max_temp = StringField('max_temp', validators=[DataRequired()])
    max_wind_speed = StringField('max_wind_speed', validators=[DataRequired()])
    avg_wind = StringField('avg_wind', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        input_data = [[
            float(form.longitude.data),
            float(form.latitude.data),
            form.month.data,
            form.day.data,
            float(form.avg_temp.data),
            float(form.max_temp.data),
            float(form.max_wind_speed.data),
            float(form.avg_wind.data)
        ]]

        input_df = pd.DataFrame(input_data, columns=[
            'longitude', 'latitude', 'month', 'day',
            'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind'
        ])

        input_prepared = full_pipeline.transform(input_df)

        model = keras.models.load_model("fires_model.keras")
        prediction = model.predict(input_prepared)
        scaled_prediction = prediction[0][0] * 10  # 단순 스케일 보정
        return render_template('result.html', prediction=np.round(scaled_prediction, 2))

    return render_template('prediction.html', form=form)


# 나중에 예측용 파이프라인 정의에 활용할 변수들
num_attribs = ['longitude', 'latitude', 'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind']
cat_attribs = ['month', 'day']
