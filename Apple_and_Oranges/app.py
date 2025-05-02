from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model/mlpAppleAndOranges.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            
            weight = float(request.form["peso"])
            size = float(request.form["tamanho"])

            input_data = scaler.transform([[weight, size]])
            result = model.predict(input_data)
            if result[0] == 1:
                fruit = "Laranja"
            else:
                fruit = "Maça"
            return render_template("result.html", fruit=fruit)
        except Exception as err:
            return f"An error occurred: {err}"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)