
import uuid

# import cv2
import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
import numpy as np
from PIL import Image
import pandas as pd

import config
import inference


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome to the PTRMS Breath Analyzer"}


@app.post("/{version}")
def get_csv(version: str, file: UploadFile = File(...)):
    mydata = np.array(pd.read_csv(file.file))
    model = config.MODELS[version]
    output  = inference.inference(model, mydata)
    name = f"/storage/{str(uuid.uuid4())}.csv"
    output.to_csv(name)
    return {"name": name}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8081)