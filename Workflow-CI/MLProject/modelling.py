import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn

def main(data_path):
    df = pd.read_csv(data_path)
    X = df.drop('Prediction', axis=1)
    y = df['Prediction']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    with mlflow.start_run():
        mlflow.sklearn.autolog()
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="emails_preprocessed.csv")
    args = parser.parse_args()
    main(args.data_path)
