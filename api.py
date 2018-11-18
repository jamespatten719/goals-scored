from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np

#{"time":10,"shot_place":2, "location":15, "assist_method":1,"bodypart":2, "situation":1}
# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if XGBgoal:
        try:
            json_ = request.json
            print(json_)

            query = pd.DataFrame(json_)
            query = query.reindex(columns=XGBgoal_columns, fill_value=0)
            #df = pd.DataFrame(columns=XGBgoal_columns)
            #query = df.append(json_, ignore_index=True)

            prediction = list(XGBgoal.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) 
    except:
        port = 12345 

    XGBgoal = joblib.load("XGBgoal.pkl") 
    print ('Model loaded')
    XGBgoal_columns = joblib.load("XGBgoal_columns.pkl") 
    print ('Model columns loaded')

    app.run(port=port, debug=True)