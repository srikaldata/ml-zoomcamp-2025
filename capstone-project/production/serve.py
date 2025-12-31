# importing necessary packages
import pickle
from pydantic import BaseModel 
from fastapi import FastAPI
import uvicorn

# create a web app
app = FastAPI(title="sales-revenue-estimation")

# class for feature data object for a record
class AdsalesFeatureset(BaseModel):
    tv: float
    radio: float
    social_media: float
    influencer: str


# class for the predictions object
class PredictResponse(BaseModel):
    sales: float


# import the pickled model
with open('pickled_model.bin', 'rb') as f_in:
    model = pickle.load(f_in)
    
# import the pickled dict vectorizer
with open('dictvectorizer_fulltrain.bin', 'rb') as vec_in:
    dict_vectorize_features = pickle.load(vec_in)

# fn for making prediction of one budget allocation across different ad channels
def predict_single(adsales_record):
    features_vectorized = dict_vectorize_features.transform(adsales_record)
    result = model.predict(features_vectorized)[0]
    return result

# web app to make sales revenue estimation
@app.post("/sales-revenue-estimate")
def predict(adsales_featureset: AdsalesFeatureset) -> PredictResponse:
    decision = predict_single(adsales_featureset.model_dump())

    return PredictResponse(
        sales = decision
    )

# accessing the model to make predictions
if __name__ == "__main__":
    uvicorn.run(app, host="#.#.#.#", port='####')
