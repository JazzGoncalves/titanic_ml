from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib

class TitanicModel:
    def __init__(self, model_path='app/titanic_model.pkl'):
        self.model_path = model_path
        self.model = RandomForestClassifier(random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        print(f"Acurácia: {accuracy:.2f}")
        print(f"F1-score: {f1:.2f}")
        print(f"Precisão: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        
        return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

    def save_model(self):
        joblib.dump(self.model, self.model_path)
        print(f"Modelo salvo em {self.model_path}")

    def load_model(self):
        self.model = joblib.load(self.model_path)
