FROM continuumio/miniconda3

WORKDIR /app

COPY conda.yaml .
RUN conda env create -f conda.yaml

SHELL ["conda", "run", "-n", "workflow-ci-env", "/bin/bash", "-c"]

COPY . .

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "workflow-ci-env", "python", "modelling.py"]
