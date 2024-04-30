from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np
import uvicorn

# Load pre-trained model
model_path = "/home/godfather101/Downloads/DBI/Web-App/TBI/TBI-Backend/Backend/Model/DBI_Model.h5"
model = load_model(model_path)

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
        prediction = model.predict(np.array([input_data]))
        result = np.argmax(prediction)
        if result == 1:
            return {"Prediction": "Abnormal Condition"}
        else:
            return {"Prediction": "Normal Condition"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")