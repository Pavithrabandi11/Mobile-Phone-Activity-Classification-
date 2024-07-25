import pickle
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = pickle.load(open(r'C:\Users\dadia\OneDrive\Desktop\ABHI\model.pkl', "rb"))
sc = pickle.load(open(r'C:\Users\dadia\OneDrive\Desktop\ABHI\scaler.pkl', "rb"))

# Load Label Encoders
dayencoder = joblib.load('DayTimeEncoder')
wkencoder = joblib.load('WeekdayEncoder')
wkndencoder = joblib.load('wkndEncoder')
cencoder = joblib.load('CostEncoder')
wencoder = joblib.load('WeatherEncoder')
nencoder = joblib.load('NameEncoder')



@app.route('/')
def Home():
    return render_template("home1.html")

@app.route('/home1',methods=['POST'])
def HOME():
    return render_template("home1.html")


@app.route('/predict1')
def Predict():
    return render_template("predict1.html")

@app.route('/about')
def About():
    return render_template('about.html')

@app.route('/contact')
def Contact():
    return render_template('contact.html')

@app.route('/predictionpage', methods=['GET','POST'])
def prediction():
    # Get form data
    df=pd.read_csv(r'C:\Users\dadia\OneDrive\Desktop\ABHI\frappe.csv')
    daytime = dayencoder.transform([request.form["daytime"]])[0]
    weekday = wkencoder.transform([request.form["weekday"]])[0]
    cost = cencoder.transform([request.form['cost']])[0]
    weather = wencoder.transform([request.form['weather']])[0]
    name = nencoder.transform([request.form['sname']])[0]

    if weekday=='sunday' or weekday=='saturday':
        iswknd='weekend'
    
    else:
        iswknd='workday'

    # Prepare the input data for prediction
    iswknd=wkndencoder.transform([iswknd])[0]
    item = df['item'].where(df['name'] == name).dropna().unique()[0]
    x_test = [[item, daytime, weekday, iswknd, cost, weather, name]]
    x_test = sc.transform(x_test)

    # Make prediction
    pred = model.predict(x_test)

    if pred[0] == 0:
        r = "Homework"
    elif pred[0] == 1:
        r = "Unknown"
    else:
        r = "Work"

    result = "The Phone activity was most likely for " + r
    return render_template("predictionpage.html",prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True,port=8000)
