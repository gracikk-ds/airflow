# -*- coding: utf-8 -*-
import os
import yaml
from airflow import DAG
from airflow.decorators import task
from datetime import timedelta, datetime

import optuna
import pandas as pd
from functools import partial

import mlflow
from mlflow.models import infer_signature
from mlflow.tracking.client import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

default_args = {
    "owner": "gracikk",
    "retries": 0,
    # "retry_delay": timedelta(minutes=2),
    "dagrun_timeout": timedelta(minutes=2),
    "catchup": False,
    "start_date": datetime(
        datetime.now().year, datetime.now().month, datetime.now().day, datetime.now().hour, datetime.now().minute
    ),
}


def objective(trial, params, x_train, y_train):
    """Run optuna trials"""
    with mlflow.start_run(nested=True):
        penalty = trial.suggest_categorical("penalty", params["penalty"])
        c = trial.suggest_float(
            "C",
            float(params["C"]["upper_bound"]),
            float(params["C"]["lower_bound"]),
            log=True,
        )

        clf = LogisticRegression(solver="liblinear", penalty=penalty, C=c)

        mlflow.log_params({"C": c, "penalty": penalty})

        target = cross_val_score(
            clf, x_train, y_train, n_jobs=-1, cv=3, scoring="precision"
        ).mean()

        mlflow.log_metrics({"train_precision": target})

    return target


def log_best_run(run, experiment_id):
    # find the best run, log its metrics as the final metrics of this run.
    client = MlflowClient()
    runs = client.search_runs(
        [experiment_id],
        "tags.mlflow.parentRunId = '{run_id}' ".format(run_id=run.info.run_id),
    )

    best_metric = 0
    best_run = None
    for r in runs:
        if r.data.metrics["train_precision"] > best_metric:
            if best_run is not None:
                run_id = best_run.info.run_id
                mlflow.delete_run(run_id)
            best_run = r
            best_metric = r.data.metrics["train_precision"]
        else:
            run_id = r.info.run_id
            mlflow.delete_run(run_id)

    mlflow.log_metrics(
        {
            "train_precision": best_metric,
        }
    )

    mlflow.log_params(
        {"C": best_run.data.params["C"], "penalty": best_run.data.params["penalty"]}
    )

    run_id = best_run.info.run_id
    mlflow.delete_run(run_id)


@task(task_id=f"find_best_params_task")
def find_best_params(**context):
    remote_server_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(remote_server_uri)

    etl_name = f"etl_v{datetime.now().year}.{datetime.now().month}.{datetime.now().day}.{datetime.now().hour}.{datetime.now().minute}"
    mlflow.set_experiment(etl_name)

    with mlflow.start_run() as run:
        """Runs training job and save best model to model's storage"""

        experiment_id = run.info.experiment_id

        # read dataset
        train = pd.read_csv("/opt/airflow/data/x_train.csv")
        # Now separate the dataset as response variable and feature variables
        x_train = train.drop("target", axis=1)
        y_train = train["target"]

        with open("/opt/airflow/artefacts/params.yaml", "r") as stream:
            params = yaml.safe_load(stream)["train"]

        obj_partial = partial(
            objective, params=params, x_train=x_train, y_train=y_train
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(obj_partial, n_trials=50)
        best_params = study.best_trial.params
        context['ti'].xcom_push(key='mlflow_experiment', value=etl_name)
        context['ti'].xcom_push(key='best_params', value=best_params)

        log_best_run(run, experiment_id)


@task(task_id="save_model_task")
def save_model(**context):
    remote_server_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(remote_server_uri)

    experiment_name = context.get("ti").xcom_pull(key="mlflow_experiment")
    # mlflow.set_experiment(mlflow_experiment)

    current_experiment = mlflow.get_experiment_by_name(experiment_name)
    df = mlflow.search_runs([current_experiment.experiment_id])
    df.sort_values(by="start_time", inplace=True)
    df.reset_index(inplace=True, drop=True)
    run_id = df.run_id.values[-1]

    with mlflow.start_run(run_id=run_id):

        # read dataset
        train = pd.read_csv("/opt/airflow/data/x_train.csv")
        # Now separate the dataset as response variable and feature variables
        x_train = train.drop("target", axis=1)
        y_train = train["target"]

        # Let's run SVC again with the best parameters.
        params = context.get("ti").xcom_pull(key="best_params")
        clf = LogisticRegression(solver="liblinear", **params)
        clf.fit(x_train, y_train)

        signature = infer_signature(x_train, y_train)
        model_name = f"model_v{datetime.now().year}.{datetime.now().month}.{datetime.now().day}.{datetime.now().hour}"
        # Log the sklearn model and register as version 1
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="sklearn-model",
            registered_model_name=model_name,
            signature=signature,
        )
        context['ti'].xcom_push(key='model_name', value=model_name)


with DAG(
    dag_id='training',
    description="training pipeline",
    schedule_interval='*/5 * * * *',
    default_args=default_args,
    tags=['train'],
    params={'stage': 'train'},
) as dag:

    second_task = find_best_params()
    third_task = save_model()

    second_task >> third_task

if __name__ == "__main__":
    dag.cli()
