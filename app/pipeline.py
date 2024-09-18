import pandas as pd
import os
from sklearn.model_selection import train_test_split
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.preprocess import TitanicDataPreprocessor
from app.model import TitanicModel

class TitanicModelPipeline:
    def __init__(self):
        self.model = TitanicModel()

    def train_model(self, df):
        print("Iniciando o processo de pré-processamento...")
        
        if 'Survived' not in df.columns:
            raise ValueError("A coluna 'Survived' não está presente no dataset.")
        
        y = df['Survived']
        df_features = df.drop(columns=['Survived'])  # Manter apenas as features

        preprocessor = TitanicDataPreprocessor()
        df_processed = preprocessor.preprocess_data(df_features)

        print("Pré-processamento concluído.")

        X_train, X_test, y_train, y_test = train_test_split(df_processed, y, test_size=0.2, random_state=42)

        print("Iniciando o treinamento do modelo...")
        self.model.train(X_train, y_train)

        print("Treinamento concluído. Avaliando o modelo...")
        self.model.evaluate(X_test, y_test)

        self.model.save_model()

        features_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'features.txt')
        print(f"Salvando as features em: {features_path}")

        with open(features_path, 'w') as f:
            for feature in X_train.columns:
                f.write(f"{feature}\n")

        print(f"Features salvas com sucesso em {features_path}")

    def predict(self, df):
        print("Iniciando o processo de previsão...")

        preprocessor = TitanicDataPreprocessor()
        df_processed = preprocessor.preprocess_data(df)

        features_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'features.txt')
        print(f"Lendo as features de: {features_path}")

        with open(features_path, 'r') as f:
            expected_columns = [line.strip() for line in f]

        for col in expected_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0

        df_processed = df_processed[expected_columns]

        print("Previsão realizada com sucesso.")
        return self.model.predict(df_processed)

if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    pipeline = TitanicModelPipeline()
    pipeline.train_model(df)
