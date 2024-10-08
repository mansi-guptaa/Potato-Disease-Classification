from fastapi import FastAPI, File, UploadFile
from enum import Enum
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image 
import tensorflow as tf


app = FastAPI()

MODEL = tf.keras.models.load_model("model/1.keras")
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


def read_file_as_image(data) -> np.ndarray:

    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file : UploadFile = File(...)
):
    image = read_file_as_image(await file.read())   
    img_batch = np.expand_dims(image, 0)
    prediction = MODEL.predict(img_batch)
    predict_class = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {'Class': predict_class, 'Confidence_Interval': float(confidence)}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)