# import required packages
import json
import requests
import pandas as pd

url = 'http://localhost:####/sales-revenue-estimate'

# import the dataframe record and extracting the feature set
inference_df = pd.read_csv('record_for_inference.csv')

feature_infer_df = inference_df.iloc[:, :4]

# convert the dataframe into a json object
feature_infer_json = json.loads(feature_infer_df.to_json(orient='records'))[0]

# payload for inference for web app served using Fast API
feature_infer_json_payload = json.dumps(feature_infer_json, separators=(', ', ':'))

# for testing -->
# what was the actual sales revenue
actual = float(inference_df.iloc[:, -1].values[0])

# json of the feature set to simulate real time json
client = feature_infer_df.to_json(orient='records')

# fetching the response from the served fast api app
response = requests.post(url, json=feature_infer_json)

# converting prediction object to json
predictions = response.json()

# printing the predictions
print()
print(response)
print()
print('prediction result:')
print(predictions)
print()
print('Actual sales revenue: $', actual)
print('Predicted sales revenue: $', round(predictions['sales'],3))
print('Difference b/w predicted and actual sales revenue: $', round(round(predictions['sales'],2)-actual, 2))
print()
