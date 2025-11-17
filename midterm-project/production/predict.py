# import required packages
import json
import requests
import pandas as pd

url = 'http://localhost:####/insulin_dosage_predict'

# import the dataframe record and extracting the feature set
inference_df = pd.read_csv('record_for_inference.csv')

feature_infer_df = inference_df.iloc[:, :10]

# convert the dataframe into a json object
feature_infer_json = json.loads(feature_infer_df.to_json(orient='records'))[0]

# payload for inference for web app served using Fast API
feature_infer_json_payload = json.dumps(feature_infer_json, separators=(', ', ':'))


# purely for testing purposes-->
# what was the actual decision
if int(inference_df.iloc[0, -1]) == 1:
    actual = ('Actual insulin dosage -->', 1, 'means \'up\'')

elif inference_df.iloc[:, -1] == 0:
    actual = ('Actual insulin dosage -->', 0, 'means \'steady\'')


# json of the feature set to simulate real time json
client = feature_infer_df.to_json(orient='records')

# fetching the response
response = requests.post(url, json=feature_infer_json)

# converting the prediction object to json
predictions = response.json()

# printing the predictions
print()
print(*actual)
print()
print(response)
print()
print('prediction result:')
print(predictions)
print()
