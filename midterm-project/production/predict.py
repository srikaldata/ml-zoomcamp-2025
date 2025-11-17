# import required packages

import requests
import pandas as pd

url = 'http://localhost:####/insulin_dosage_predict'


# import the dataframe record and extracting the feature set
inference_df = pd.read_csv('record_for_inference.csv')

feature_infer_df = inference_df.iloc[:, :10]


# purely for testing purposes-->
# what was the actual decision
if inference_df.iloc[:, -1] == 1:
    actual = (1, 'up')

elif inference_df.iloc[:, -1] == 0:
    actual = (0, 'steady')


# json of the feature set to simulate real time json
client = feature_infer_df.to_json(orient='records')


# fetching the response
response = requests.post(url, json=client)

# converting the prediction object to json
predictions = response.json()

# printing the predictions 
print(response, predictions, actual)
