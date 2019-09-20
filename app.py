from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from model import NLPModel

app = Flask(__name__)
api = Api(app)

model = NLPModel()
import sqlite3
'''
import sqlite3

conn = sqlite3.connect('database.db')

conn.execute('CREATE TABLE students (Input TEXT, Prediction TEXT)')
conn.close()
'''
clf_path = r'path of the folder where your pickle files are stored/FitaraClassifier.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)

vec_path = r'path of the folder where your pickle files are stored/CountVectorizer.pkl'
with open(vec_path, 'rb') as f:
    model.vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/list')
def list():
   con = sqlite3.connect("database.db")
   con.row_factory = sqlite3.Row
   
   cur = con.cursor()
   cur.execute("select * from students")
   
   rows = cur.fetchall(); 
   return render_template("list.html",rows = rows)

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = request.form.values()
    uq_vectorized = model.vectorizer_transform(features)
    prediction = model.predict(uq_vectorized)
    pred_proba = model.predict_proba(uq_vectorized)
    # Output either 'Negative' or 'Positive' along with the score
    if prediction == 0:
            pred_text = 'is not'
    else:
            pred_text = 'is'

    # round the predict proba value and set to new variable
    confidence = round(pred_proba[0], 3)
    # create JSON object
    output = pred_text
    input1 = request.form['input']
    with sqlite3.connect("database.db") as con:
            cur = con.cursor()
            cur.execute("INSERT INTO students (Input,Prediction) VALUES (?,?)",(input1,output) )
    return render_template('index.html', prediction_text='The entered text {} Fitara text.'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
