from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

classes=pickle.load(open('models/crop_classes.pkl','rb'))
print(classes)

@app.route('/')
def dashboard():
     return render_template('index.html')

@app.route('/crop_recommender')
def data_input():
    return render_template('crop_recommender.html')

@app.route('/recform', methods=['POST'])
def regform():
    N = request.form['N']
    P = request.form['P']       #Selected contact id type
    K = request.form['K']
    temperature = request.form['temperature'] 
    humidity = request.form['humidity']
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