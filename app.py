from flask import Flask, jsonify, request, render_template
import joblib
import pandas as pd
app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    test_json = request.get_json()
    dataJson = { 'age': [test_json['age']],'sex': [test_json['sex']], 'cp': [test_json['cp']], 'fbs': [test_json['fbs']], 'trtbps': [
        test_json['trtbps']],     'chol': [test_json['chol']],   'thalachh': [test_json['thalachh']],   'oldpeak': [test_json['oldpeak']], 
        'restecg': [test_json['restecg']], 'exng': [test_json['exng']], 'slp': [test_json['slp']], 'caa': [test_json['caa']], 'thall': [test_json['thall']]}

    data = pd.DataFrame(dataJson)
    log_reg = joblib.load("./log_reg_model.pkl")
    prediction = log_reg.predict(data)
    classes = ['No Heart Attack', 'Heart Attack']
    text = classes[prediction[0]]
    return jsonify({"result": text})


if __name__ == '__main__':
    app.run(debug=False)