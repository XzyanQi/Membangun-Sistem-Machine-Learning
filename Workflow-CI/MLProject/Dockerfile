FROM continuumio/miniconda3

WORKDIR /app

COPY conda.yaml .

RUN conda env create -f conda.yaml

SHELL ["conda", "run", "-n", "workflow-ci-env", "/bin/bash", "-c"]

COPY . .

ENV MLFLOW_TRACKING_URI=file:///app/mlruns
ENV PYTHONPATH=/app

RUN conda run -n workflow-ci-env python -c "import mlflow; print('MLflow version:', mlflow.__version__)"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "workflow-ci-env"]
CMD ["python", "modelling.py", "--data_path", "emails_preprocessed.csv"]
