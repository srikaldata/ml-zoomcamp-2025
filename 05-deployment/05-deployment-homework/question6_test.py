# import required packages

import requests

url = 'http://localhost:####/question6_predict'


client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}


# fetching the response
response = requests.post(url, json=client)

# converting the prediction object to json
predictions = response.json()

# printing the predictions 
print(response, predictions)
