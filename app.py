from flask import Flask, render_template, request, redirect, Markup
import pickle
import numpy as np
import config
import requests
import config
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9

app = Flask(__name__)

classes=pickle.load(open('utils/crop_classes.pkl','rb'))
disease_classes = pickle.load(open('utils/disease_classes.pkl','rb'))
disease_desc = pickle.load(open('utils/disease_desc.pkl','rb'))
disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

def predict_image(img, model=disease_model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    return prediction

def weather_fetch(city_name):
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

@app.route('/diesease', methods=['GET','POST'])
def app_data():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            print("File not accessible")
            return render_template('disease.html')
        try:
            img = file.read()
            prediction = predict_image(img)
            prediction = Markup(str(disease_desc[prediction]))
            return render_template('disease-result.html', predictions=prediction)
        except:
            pass
    return render_template('disease.html')

    
@app.route('/recform', methods=['POST'])
def regform():
    N = request.form['N']
    P = request.form['P']       
    K = request.form['K']
    city = request.form['city']
    temperature, humidity = weather_fetch(city)
    ph = request.form['ph']       
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