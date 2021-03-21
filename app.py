from flask import Flask, render_template, url_for, request, redirect
import numpy as np
import requests
from flask_cors import cross_origin
import sklearn
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
cancer_model  = pickle.load(open('cancer_model.pkl',"rb"))
diabetes_model  = pickle.load(open('diabetes_model.pkl',"rb"))
heart_model  = pickle.load(open('heart.pkl',"rb"))
liver_model  = pickle.load(open('liver.pkl',"rb"))
@app.route("/")
def home():
	return render_template("home.html")

@app.route("/cancer",methods=['POST','Get'])
def cancer():
	if request.method == "POST":
		vals = request.form.to_dict()
		vals = list(map(float,list(vals.values())))
		prediction = cancer_model.predict([vals])[0]
		if prediction == "B":
			text = 'You have B type cancer'
		else:
			text = "You have M type cancer"
		return render_template('cancer.html',context = text)
	else:
		return render_template("cancer.html",context = '')

@app.route("/liver",methods=['POST','Get'])
def liver():
	if request.method == "POST":
		vals = request.form.to_dict()
		vals = list(map(float,list(vals.values())))
		prediction = liver_model.predict([vals])[0]
		if prediction == "1":
			text = 'You have a Liver problem! Take care'
		else:
			text = "You have a great Liver!!"
		return render_template('liver.html',context = text)
	else:
		return render_template("liver.html",context = '')

@app.route("/heart",methods=['POST','Get'])
def heart():
	if request.method == "POST":
		vals = request.form.to_dict()
		vals = list(map(float,list(vals.values())))
		prediction = heart_model.predict([vals])[0]
		if prediction == "1":
			text = 'You have Heart problem! Take care'
		else:
			text = "You have a great heart!!"
		return render_template('heart.html',context = text)
	else:
		return render_template("heart.html",context = '')

@app.route("/diabetes",methods=['POST','Get'])
def diabetes():
	if request.method == "POST":
		vals = request.form.to_dict()
		vals = list(map(float,list(vals.values())))
		prediction = diabetes_model.predict([vals])[0]
		if prediction == "1":
			text = 'You have Diabetes! Take care'
		else:
			text = "You donot have diabetes"
		return render_template('diabetes.html',context = text)
	else:
		return render_template("diabetes.html",context = '')
		
if __name__ == "__main__":
	app.run(debug=True)