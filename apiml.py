from typing import Union
import uvicorn
import numpy as np
import pandas as pd
import pickle as pk
from sklearn.linear_model import LogisticRegression
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

class ScoringItem(BaseModel):
    qs1: int
    qs2: int
    qs3: int
    qs4: int
    qs5: int
    qs6: int
    qs7: int
    qs8: int


with open('mlmodel.pkl','rb') as f:
    model = pk.load(f)

@app.post("/")
async def func(item:ScoringItem):
    X_new = np.array([list(item.dict().values())])
    yhat = model.predict(X_new)[0]
    return {"prediction":str(yhat)}