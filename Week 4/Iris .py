# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os

# Set the working directory to a known existing path
os.chdir('/Users/mariapinto/Downloads')  # Ensure this directory exists


# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Save the model
model_filename = 'iris_decision_tree_model.joblib'
joblib.dump(clf, model_filename)

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('iris_decision_tree_model.joblib')

@app.route('/')
def home():
    return "Iris Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict(np.array(data['features']).reshape(1, -1))
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)

# Training and saving the model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Save the model
model_filename = 'iris_decision_tree_model.joblib'
joblib.dump(clf, model_filename)

# Flask app to serve the model
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load(model_filename)

@app.route('/')
def home():
    return "Welcome to the Iris Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict(np.array(data['features']).reshape(1, -1))
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)

# Generating the PDF document
from fpdf import FPDF
import matplotlib.pyplot as plt
from sklearn import tree

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Model Deployment Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page %s' % self.page_no(), 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

pdf = PDF()
pdf.add_page()

# Add the details
pdf.set_font('Arial', 'B', 16)
pdf.cell(0, 10, 'Name: Maria Pinto', 0, 1)
pdf.cell(0, 10, 'Batch Code: LISUM33', 0, 1)
pdf.cell(0, 10, 'Submission Date: 2024-06-01', 0, 1)
pdf.cell(0, 10, 'Submitted to: Data Glacier', 0, 1)
pdf.ln(10)

# Chapter 1: Data Loading and Training
pdf.chapter_title('Chapter 1: Data Loading and Training')
pdf.chapter_body('Loaded the Iris dataset from sklearn. Split the data into training and testing sets, and trained a Decision Tree classifier.')

# Save a snapshot of the model
fig, ax = plt.subplots(figsize=(10, 8))
tree.plot_tree(clf, filled=True)
fig.savefig('decision_tree.png')
plt.close(fig)

pdf.image('decision_tree.png', x=10, y=None, w=190)

# Chapter 2: Model Saving
pdf.chapter_title('Chapter 2: Model Saving')
pdf.chapter_body('Saved the trained model using joblib.')

# Chapter 3: Flask Deployment
pdf.chapter_title('Chapter 3: Flask Deployment')
pdf.chapter_body('Created a Flask web application to serve the model. The API has two endpoints: \n1. Home - returns a welcome message.\n2. Predict - takes a JSON input with features and returns the prediction.')

# Save the PDF
pdf.output('Model_Deployment_Report.pdf')

