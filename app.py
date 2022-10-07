from flask import Flask,render_template,url_for,request,jsonify
import pandas as pd 
import pickle
import numpy as np


app = Flask(__name__)
rf_model=pickle.load(open('spam-sms-rfmodel.pkl','rb'))
tfidf=pickle.load(open('tfidf_transform.pkl','rb'))

@app.route('/')
def home():
	return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    message = request.form['message']
    data = [message]
    vect = tfidf.transform(data).toarray()
    my_prediction = rf_model.predict(vect)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
