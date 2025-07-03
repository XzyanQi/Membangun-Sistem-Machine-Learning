import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import mlflow.sklearn

df = pd.read_csv('emails_preprocessed.csv')
X = df.drop('Prediction', axis=1)
y = df['Prediction']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run(run_name="logreg_autolog_local"):
    mlflow.sklearn.autolog()
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    pd.DataFrame(cm).to_csv("confusion_matrix.csv", index=False)
    mlflow.log_artifact("model.pkl")
    mlflow.log_artifact("confusion_matrix.csv")
