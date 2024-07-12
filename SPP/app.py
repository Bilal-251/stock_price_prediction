from flask import Flask,render_template, request
import joblib
import pandas as pd
app = Flask(__name__)

model=joblib.load('spp.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    open_price = float(request.form['open'])
    high_price = float(request.form['high'])
    low_price = float(request.form['low'])

    features = pd.DataFrame([[open_price, high_price, low_price]], columns=['Open', 'High', 'Low'])

    prediction = model.predict(features)
    formatted_prediction = f"{prediction[0]:.2f}"

    return render_template('index.html', prediction_text='Predicted Value: {}'.format(formatted_prediction))

if __name__ == '__main__':
    app.run(debug=True)