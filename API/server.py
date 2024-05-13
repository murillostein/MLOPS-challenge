# File to write model handler functions (Load model, preprocess, do inferences, ...)

from fastapi import FastAPI, Request
import pickle
import numpy as np

app_name = 'RFC model'
app_version = '1.0'
app = FastAPI(
    title=app_name,
    description='RFC model deploy',
    version=app_version
)

# Load model
def initialize():
    with open('rfc.pkl', 'rb') as model_file:
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
    #input = np.array(body['digit_image']).reshape(1, len(body['digit_image']))
    input = np.array(body['digit_image']).reshape(1, -1)
    digit_value = model.predict(input)[0].item()

    return {'digit_value': digit_value}

@app.post('/batch_predict')
async def batch_predict(request: Request):
    body = await request.json()
    inputs = np.array(body['digit_images'])
    digit_values = model.predict(inputs).tolist() 
    return {'digit_values': digit_values}