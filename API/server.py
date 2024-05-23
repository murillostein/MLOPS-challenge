# File to write model handler functions (Load model, preprocess, do inferences, ...)
from fastapi import FastAPI, Request
import pickle
import numpy as np

app_name = 'Model deploy MS'
app_version = '1.0'
app = FastAPI(
    title=app_name,
    description='Simple model deploy example',
    version=app_version
)

# Load model
def initialize():
    with open('model_logreg.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    return model
model = initialize()

@app.get('/')
async def home_page():
    """Check app health"""
    return{app_name:app_version}

@app.post('/single_predict')
async def single_predict(request: Request):
    body = await request.json()
    input = np.array(body['iris_image']).reshape(1, len(body['iris_image']))
    digit_value = model.predict(input)[0].item()

    return {'digit_value': digit_value}
