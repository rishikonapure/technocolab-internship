from flask import Flask, render_template, request
import pickle
import numpy as np


model = pickle.load(open('model/parkinsons.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('web.html')


@app.route('/predict', methods=['POST'])
def home():

    features= np.array([[float(x) for x in request.form.values()]])
     
    pred = model.predict(features)
    
    if(pred[0]==0):
    	a="Healthy"
    else:
    	a= "Not Healthy"
    return ( "Output is : "+ a )


if __name__ == "__main__":
    app.run(debug=True)