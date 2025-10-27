# uv virtual environment with sklearn and needed libraries from Q1 and Q2 have been setup

import pickle

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)

# QUESTION 3
# code to score a single record
temp_y = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

print("probability that a given lead in Q3 will convert:", predict_single(temp_y))

# run uv run python 13_predict.py in the virtual environment to find the probability
# the probability of the given lead to convert was predicted to be 0.533 (approx)
