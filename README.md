# Titanic Survival Prediction API

Este projeto consiste em uma API para prever a sobrevivência de passageiros do Titanic com base em diferentes características, como classe, idade, sexo, etc. O modelo de machine learning foi treinado utilizando o dataset Titanic da competição do Kaggle, e a API foi construída usando FastAPI.

## Recursos Principais
- **Machine Learning Model**: Um modelo de classificação treinado para prever a sobrevivência de passageiros.
- **API RESTful**: Implementada com FastAPI para expor o modelo através de endpoints HTTP.
- **Docker**: O projeto inclui um arquivo Dockerfile para fácil conteinerização e implantação.

## Requisitos
- Python 3.7+
- FastAPI
- Scikit-learn
- Pandas
- Uvicorn
- Docker (opcional para execução em container)

  ## Instalação Local
  Clone o repositório:
   git clone https://github.com/seu_usuario/titanic_ml.git
   cd titanic_ml

  ## Instale as dependências
  pip install -r requirements.txt

  ## Inicie a aplicação localmente
  uvicorn app.main:app --reload

  ## Acesse a API em: http://127.0.0.1:8000

  ## Endpoint para previsão
  * POST /predict: Envia dados de passageiros para obter a previsão de sobrevivência.
 
  **Exemplos de Uso**
{
    "Pclass": 3,
    "Age": 22.0,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Sex_male": 1,
    "Embarked_Q": 0,
    "Embarked_S": 1,
    "FamilySize": 2,
    "IsAlone": 1,
    "Cabin_U": 1,
    "TicketPrefix_NoPrefix": 1,
    "Title_Miss": 0,
    "Title_Mrs": 0,
    "Title_Other": 1
}




