# imoprting packages
import pickle
from pydantic import BaseModel 
from fastapi import FastAPI
import uvicorn

# create a web app
app = FastAPI(title="lead-conversion")

# class for inpu lead feature data object
class LeadFeatureset(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# class for the predictions object
class PredictResponse(BaseModel):
    conversion_probability: float
    convert: bool

# open the model
with open('pipeline_v2.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# fn for making a unit prediction
def predict_single(lead):
    result = pipeline.predict_proba(lead)[0, 1]
    return float(result)

# web app to make lead predictions
@app.post("/question6_predict")
def predict(lead_featureset: LeadFeatureset) -> PredictResponse:
    prob = predict_single(lead_featureset.model_dump())

    return PredictResponse(
        conversion_probability=prob,
        convert=prob >= 0.5
    )

# accessing the model to make predictions
if __name__ == "__main__":
    uvicorn.run(app, host="#.#.#.#", port='####')
