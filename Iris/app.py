from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model/mlpIris.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])
            
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            input_scaled = scaler.transform(input_data)
            
            prediction = model.predict(input_scaled)
            species = prediction[0] 


            return render_template("result.html", species=species)
        except Exception as err:
            return f"An error occurred: {err}"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)