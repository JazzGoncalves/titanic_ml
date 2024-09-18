from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import TitanicModel
import pandas as pd

app = FastAPI()

class PassengerData(BaseModel):
    Pclass: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Sex_male: int
    Embarked_Q: int
    Embarked_S: int
    FamilySize: int
    IsAlone: int
    Cabin_U: int
    TicketPrefix_NoPrefix: int
    Title_Miss: int
    Title_Mrs: int
    Title_Other: int

@app.get("/")
def root():
    return {"message": "Titanic Survival Prediction API"}

@app.post("/predict")
def predict_survival(data: PassengerData):
    model = TitanicModel()
    model.load_model()

    df = pd.DataFrame([data.model_dump()])

    expected_columns = [
        'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone',
        'Sex_male', 'Embarked_Q', 'Embarked_S', 'Title_Miss', 'Title_Mrs',
        'Title_Other', 'TicketPrefix_NoPrefix', 'Cabin_U'
    ]

    df = df[expected_columns]

    try:
        prediction = model.predict(df)
        return {"survived": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))