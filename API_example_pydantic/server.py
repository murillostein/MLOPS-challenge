from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app_name = "Shipping Price Estimator"
app_version = "1.0"

app = FastAPI(title = app_name,
              description ="Estimate the cost of shipping a specific product",
              version = app_version)

# Load Model
infile = open('shipping_estimate_model.pkl', 'rb')
model = pickle.load(infile)
infile.close()

class Model_input(BaseModel):
    """Input for data validation"""
    price: float 
    product_weight_g: float
    product_height_cm: float
    delivery_distance_km: float
    product_volume_cm3: float


class Output(BaseModel):
    """Ouput for data validation"""
    shipping_estimated_price: float


@app.get('/')
async def home_page():
    """Check app health"""
    return{app_name:app_version}

@app.post("/predict", response_model=Output)
async def make_predction(input: Model_input):
    """Make predictions"""
    input_df = pd.DataFrame(input.dict(),index = [0])
    predict_price = model.predict(input_df)[0]
    return {'shipping_estimated_price': round(predict_price,2)}
