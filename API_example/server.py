from fastapi import FastAPI, Request
import pickle
import numpy as np

app_name = 'Simple Model'
app_version = '1.0'
app = FastAPI(
    title=app_name,
    description='Simple model deploy example',
    version=app_version
)

# Load model
def initialize():
    with open('simple_model.pkl', 'rb') as model_file:
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
    input = np.array(body['digit_image']).reshape(1, len(body['digit_image']))
    digit_value = model.predict(input)[0].item()

    return {'digit_value': digit_value}





