from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/result')
def result():
    # ... get the result ...
    return render_template('index.html', result=result)
