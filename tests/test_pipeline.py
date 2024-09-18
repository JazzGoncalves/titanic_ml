import sys
import os
import pandas as pd
from app.pipeline import TitanicModelPipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.pipeline import TitanicModelPipeline


def test_pipeline():

    df_test = pd.read_csv('data/test.csv')

    pipeline = TitanicModelPipeline()

    pipeline.model.load_model()

    predictions = pipeline.predict(df_test)

    assert len(predictions) == len(df_test), "O número de previsões não corresponde ao número de amostras"
    print("Teste de pipeline bem-sucedido!")
