from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import re
from typing import List

app = FastAPI()

with open('model.pkl', 'rb') as handle:
    model = pickle.load(handle)

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float

def preprocess_data(df: pd.DataFrame):
    df = df.copy()
    df['name'] = df['name'].apply(lambda x: x.split()[0])
    df['seats'] = df['seats'].astype(str)
    df.reset_index(drop=True, inplace=True)
    df['mileage'] = df['mileage'].apply(lambda x: float(re.findall('\d+[.\s]*\d+', str(x).replace(',', '.'))[0]) if type(x) == str else None)
    df['engine'] = df['engine'].apply(lambda x: float(re.findall('\d+[.\s]*\d+', str(x).replace(',', '.'))[0]) if type(x) == str else None)
    df['max_power'] = df['max_power']\
    .apply(lambda x: (''.join([c for c in x if not (c.isalpha() or c==' ')])) if type(x) == str else x).apply(lambda x: float(x) if x != '' else None)
    parsed = df['torque'].apply(lambda x: re.findall('\d+[.\s]*\d+', str(x).replace(',', '.')))
    parsed_dim = df['torque'].apply(lambda x: re.search('([a-zA-Z]*gm|[a-zA-Z]*nm)', str(x).lower())).apply(lambda x: x.group(0) if x is not None else None)
    df['torque'] = parsed.apply(lambda x: float(x[0]) if len(x) else None) * parsed_dim.apply(lambda x: 9.8 if x == 'kgm' else 1)
    df['max_torque_rpm'] = parsed.apply(lambda x: float(x[-1].replace('.','')) if len(x) else None)

    df['age'] = (2022 - df['year'])
    df['driven per year'] = df['km_driven'] / (df['age'])
    df['dealer+first'] = ((df['seller_type'] == 'Dealer') & (df['owner']=='First Owner')).astype(str)
    df['mean_horses'] = df['max_power'] / df['engine']
    df['squared_age'] = df['age']**2
    df.drop('year', axis=1, inplace=True)
    df['km_driven'] = np.log1p(df['km_driven'])
    df['driven per year'] = np.log1p(df['driven per year'])
    df['was_na'] = np.any(df.isna(), axis=1).astype(str)
    return df.drop('selling_price', axis=1)

def make_df(items: List[Item]):
    return pd.DataFrame.from_records([i.dict() for i in items])

class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = make_df([item])
    print(df)
    X = preprocess_data(df)
    pred = model.predict(X)
    return {'your_price': np.expm1(pred[0])}

