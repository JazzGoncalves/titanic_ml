import pytest
import pandas as pd
from app.model import TitanicModel
from app.preprocess import TitanicDataPreprocessor
import os

@pytest.fixture
def model():
    model = TitanicModel()
    model.load_model()
    return model

def test_model_loading_and_prediction(model):
    df_test = pd.read_csv('data/test.csv')
    preprocessor = TitanicDataPreprocessor()
    df_processed = preprocessor.preprocess_data(df_test)

    features_path = 'app/features.txt'
    with open(features_path, 'r') as f:
        expected_columns = [line.strip() for line in f]

    for col in expected_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0

    df_processed = df_processed[expected_columns]

    predictions = model.predict(df_processed)
    assert len(predictions) == len(df_test), "Número de previsões não corresponde ao número de amostras"

def test_model_training_and_saving():

    df_train = pd.read_csv('data/train.csv')
    preprocessor = TitanicDataPreprocessor()
    df_processed = preprocessor.preprocess_data(df_train)

    X = df_processed.drop(columns=['Survived'])
    y = df_train['Survived']

    model = TitanicModel()
    model.train(X, y)
    model.save_model()

    assert os.path.exists('app/titanic_model.pkl'), "O modelo não foi salvo corretamente"
