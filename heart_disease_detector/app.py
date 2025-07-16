from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load or train the model
try:
    model = pickle.load(open("model/heart_model.pkl", "rb"))
except:
    df = pd.read_csv("dataset/dataset.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    pickle.dump(model, open("model/heart_model.pkl", "wb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = [float(request.form[key]) for key in request.form]
        prediction = model.predict([data])
        result = "Heart Disease Detected 💔" if prediction[0] == 1 else "No Heart Disease ❤️"
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
