from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        tv = float(request.form["tv"])
        radio = float(request.form["radio"])
        newspaper = float(request.form["newspaper"])

        input_data = pd.DataFrame({
            "TV": [tv],
            "Radio": [radio],
            "Newspaper": [newspaper]
        })

        prediction = model.predict(input_data)[0]
        prediction = round(prediction, 2)

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)