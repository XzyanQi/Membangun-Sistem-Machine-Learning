from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from prometheus_client import start_http_server, Counter, Gauge
import time

model = pickle.load(open("model_tuning.pkl", "rb"))


prediction_counter = Counter('total_predictions', 'Total number of predictions')
prediction_latency = Gauge('prediction_latency_seconds', 'Latency of predictions')
error_counter = Counter('prediction_errors', 'Total prediction errors')

app = FastAPI()

class EmailFeatures(BaseModel):
    features: list[float]  

@app.post("/predict")
def predict(data: EmailFeatures):
    start_time = time.time()
    try:
        arr = np.array(data.features).reshape(1, -1)  
        prediction = model.predict(arr)
        latency = time.time() - start_time
        prediction_counter.inc()
        prediction_latency.set(latency)
        return {"prediction": int(prediction[0]), "latency": latency}
    except Exception as e:
        error_counter.inc()
        return {"error": str(e)}

start_http_server(8001)
