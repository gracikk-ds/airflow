# -*- coding: utf-8 -*-
from airflow import DAG
from airflow.decorators import task
from datetime import timedelta, datetime
import pandas as pd
import mlflow
from mlflow.tracking.client import MlflowClient


default_args = {
    "owner": "gracikk",
    "retries": 0,
    "dagrun_timeout": timedelta(minutes=2),
    "catchup": False,
    "start_date": datetime(
        datetime.now().year, datetime.now().month, datetime.now().day, datetime.now().hour, datetime.now().minute
    ),
}


# @task(task_id="get_run_id")
# def get_run_id(**context):
#     client = MlflowClient()
#     experiments = client.list_experiments()
#     current_experiment = experiments[-1]
#     df = mlflow.search_runs([current_experiment.experiment_id])
#     df.sort_values(by="start_time", inplace=True)
#     df.reset_index(inplace=True, drop=True)
#     run_id = df.run_id.values[-1]
#     context['ti'].xcom_push(key='run_id', value=run_id)


@task(task_id="batch_prediction")
def predict(**context):
    """Runs model's predict method"""

    path_to_dataset = "/opt/airflow/data/x_test.csv"
    path_to_predictions_storage = "/opt/airflow/data/"

    # read dataset
    x_test = pd.read_csv(path_to_dataset)
    if "target" in x_test.columns:
        x_test = x_test.drop("target", axis=1)

    client = MlflowClient()
    registered_model_name = context['ti'].xcom_pull(
        dag_id='training', task_ids='save_model_task', key="model_name", include_prior_dates=True
    )
    latest_version = client.get_latest_versions(
        registered_model_name, stages=["None"]
    )[0].version

    clf = mlflow.sklearn.load_model(
        f"models:/{registered_model_name}/{latest_version}"
    )
    predictions = pd.DataFrame(data=clf.predict(x_test), columns=["wine_quality"])
    predictions.to_csv(f"{path_to_predictions_storage}/predictions.csv")


with DAG(
    dag_id='prediction',
    description="prediction",
    schedule_interval='*/5 * * * *',
    default_args=default_args,
    tags=['prediction'],
    params={'stage': 'prediction'},
) as dag:

    second_task = predict()

    second_task

if __name__ == "__main__":
    dag.cli()
