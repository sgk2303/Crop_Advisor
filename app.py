from flask import Flask, render_template, request
import pickle
import numpy as np
import config
import requests

app = Flask(__name__)

classes=pickle.load(open('models/crop_classes.pkl','rb'))
print(classes)

def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    print(city_name)
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


@app.route('/')
def dashboard():
     return render_template('index.html')

@app.route('/crop_recommender')
def data_input():
    return render_template('crop_recommender.html')

@app.route('/diesease')
def app_data():
    return render_template('diesease.html')
    
@app.route('/recform', methods=['POST'])
def regform():
    N = request.form['N']
    P = request.form['P']       #Selected contact id type
    K = request.form['K']
    city = request.form['city']
    print(city)
    temperature, humidity = weather_fetch(city)
    ph = request.form['ph']       #Selected contact id type
    rainfall = request.form['rainfall']
    data=[[N,P,K,temperature,humidity,ph,rainfall]]
    model=pickle.load(open('models/crop_recomender.pkl','rb'))
    pred = model.predict(data)
    max_prob=max(pred[0])
    max_prob
    class_index = (np.where(pred[0]==max_prob)[0][0])
    res = classes[class_index]
    return render_template('crop_result.html',crop=res)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)