import pickle
import pandas as pd
from flask import Flask, render_template, request

model = pickle.load(open('model.pkl', 'rb'))

categorical_columns = ['Cat12', 'Cat10', 'Cat8', 'Cat6', 'Blind_Model']
numerical_columns = ['NVVar3', 'Var3', 'Var7', 'Var2', 'Var1']

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            form_data = request.form.to_dict()

            for col in numerical_columns:
                form_data[col] = float(form_data[col])

            for col in categorical_columns:
                form_data[col] = hash(form_data[col]) % 1000

            input_data = pd.DataFrame([form_data])

            input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

            prediction = model.predict(input_data)

            return f"Prediction: {prediction[0]}"
        except Exception as e:
            return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
