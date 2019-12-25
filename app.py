import numpy as np 
from flask import Flask,request,jsonify,render_template
import pickle

app = Flask(__name__)
with open('model_fromscript','rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def home2():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('index.html', prediction_text = 'Expected result is ' + prediction)

if __name__ == "__main__":
    app.run(debug=True)