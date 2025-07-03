import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn

df = pd.read_csv('emails_preprocessed.csv')
X = df.drop('Prediction', axis=1)
y = df['Prediction']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear']
}

gs = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    param_grid, cv=3, scoring='f1'
)
gs.fit(X_train, y_train)
best_model = gs.best_estimator_

y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm).to_csv("confusion_matrix_tuning.csv", index=False)

with open("model_tuning.pkl", "wb") as f:
    pickle.dump(best_model, f)

with mlflow.start_run(run_name="tuning_manual_logging_local"):
    mlflow.log_param('C', best_model.C)
    mlflow.log_param('solver', best_model.solver)
    mlflow.log_metric('accuracy', accuracy_score(y_test, y_pred))
    mlflow.log_metric('precision', precision_score(y_test, y_pred, zero_division=0))
    mlflow.log_metric('recall', recall_score(y_test, y_pred, zero_division=0))
    mlflow.log_metric('f1_score', f1_score(y_test, y_pred, zero_division=0))

    mlflow.sklearn.log_model(best_model, "model_tuned")
    mlflow.log_artifact("model_tuning.pkl")
    mlflow.log_artifact("confusion_matrix_tuning.csv")

    print("Best Params:", gs.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", cm)
