# predict.py
from crypt import methods
from ctypes.wintypes import PINT
import re
import pandas as pd
import numpy as np
import pickle
from train import preprocess_data
from flask import Flask
from flask import request
from flask import jsonify


# Load the preprocessing function and model
with open('rf_model.bin', 'rb') as file:
    dv, best_model = pickle.load(file)

print("Dv and Model loaded successfully!")

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    flight = request.get_json()

    flight = pd.DataFrame([flight])
    # Apply preprocessing
    X_new = preprocess_data(flight,
                            drop_columns=['Unnamed: 0', 'flight'],
                            log_columns=['duration'],
                            outlier_removal=True)

    X_dict = X_new.to_dict(orient='records')
    X = dv.transform(X_dict)

    # Make predictions
    # cancel the log effect
    price_pred = np.expm1(best_model.predict(X))
    result = {
        'Predicted_Price': price_pred.tolist()
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
