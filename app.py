import pickle
import pandas as pd
from flask import Flask,render_template,request

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route("/" ,methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            form_data = {
                "NVVar3": float(request.form['NVVar3']),
                "Var3": float(request.form['Var3']),
                "Var7": float(request.form['Var7']),
                "Cat12": request.form['Cat12'],
                "Cat10": request.form['Cat10'],
                "Cat8": request.form['Cat8'],
                "Var2": float(request.form['Var2']),
                "Var1": float(request.form['Var1']),
                "Cat6": request.form['Cat6'],
                "Blind_Model": request.form['Blind_Model']
            }
            input_data = pd.DataFrame([form_data])
            prediction = model.predict(input_data)
            return f"Prediction: {prediction[0]}"
        except ValueError as ve:
            return f"Value Error: {ve}"
        except Exception as e:
            return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)