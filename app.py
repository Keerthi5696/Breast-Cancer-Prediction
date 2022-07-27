import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__, static_url_path='/static')
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('cancer.html')


@app.route('/predict', methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    y_probabilities_test = model.predict_proba(final_features)
    y_prob_success = y_probabilities_test[:, 1]
    print("final features", final_features)
    print("prediction:", prediction)
    output = (prediction[0])
    y_prob = round(y_prob_success[0], 3)
    print(output)

    if output == 0:
        return render_template('cancer.html', prediction_title='BENIGN CANCER', prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A BENIGN CANCER....The probability value is  {}'.format(y_prob))
    else:
        return render_template('cancer.html', prediction_title='MALIGNANT CANCER', prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A MALIGNANT CANCER.... The probability value is  {}'.format(y_prob))


@app.route('/predict_api', methods=['POST'])
def predict_api():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
