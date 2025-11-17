# importing necessary packages
import pickle
from pydantic import BaseModel 
from fastapi import FastAPI
import uvicorn

# create a web app
app = FastAPI(title="insulin-dosage-decision")

# class for input lead feature data object
class PatientrecordFeatureset(BaseModel):
    family_history: str
    glucose_level: float
    physical_activity: float
    previous_medications: str
    bmi: float
    hba1c: float
    weight: float
    insulin_sensitivity: float
    sleep_hours: float
    creatinine: float


# class for the predictions object
class PredictResponse(BaseModel):
    insulin: int


# open the model
with open('pickled_model.bin', 'rb') as f_in:
    model = pickle.load(f_in)
    
# open the dict vectorizer
with open('dictvec_fulltrain.bin', 'rb') as vec_in:
    dict_vectorize_feature = pickle.load(vec_in)

# fn for making prediction of one patient health record 
def predict_single(patient_health_record):
    features_vectorized = dict_vectorize_feature.transform(patient_health_record)
    result = model.predict(features_vectorized)
    return float(result)

# web app to make insulin dosage decision
@app.post("/insulin_dosage_predict")
def predict(patientrecord_featureset: PatientrecordFeatureset) -> PredictResponse:
    decision = predict_single(patientrecord_featureset.model_dump())

    return PredictResponse(
        insulin= decision
    )

# accessing the model to make predictions
if __name__ == "__main__":
    uvicorn.run(app, host="#.#.#.#", port='####')
