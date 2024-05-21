from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np
import uvicorn
import pickle

# Load pre-trained model
model_path = "model_TBI.keras"
model = load_model(model_path)

# scaler

with open("scaler.pkl","rb") as file:
    scaler = pickle.load(file)

# Define FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request body
class InputData(BaseModel):
    data: list

# Home route
@app.get("/")
async def home():
    return {"model": "DBI"}

# Prediction endpoint
@app.post("/api")
async def predict(data: InputData):
    try:
        input_data = data.data
        prediction = model.predict(scaler.transform(np.array(input_data).reshape(1,-1)))
        if np.argmax(prediction) == 0:
            return {"Prediction": "normal Condition"}
        else:
            return {"Prediction": "abnormal Condition"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, log_level="info")
