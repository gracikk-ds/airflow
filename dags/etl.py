import pandas as pd
from datetime import timedelta, datetime
from airflow import DAG
from airflow.decorators import task
# from sklearn.preprocessing import LabelEncoder, StandardScaler
from dask_ml.preprocessing import LabelEncoder, StandardScaler
from dask_ml.model_selection import train_test_split
from dask.dataframe import read_csv
# from sklearn.model_selection import train_test_split


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


@task(task_id="extract_task")
def extract(**context):
    # Making binary classificaion for the response variable.
    # Dividing wine as good and bad by giving the limit for the quality
    bins = (2, 6.5, 8)
    group_names = ["bad", "good"]

    wine = read_csv("https://drive.google.com/uc?id=1CmuwSopTdpw-4rYq57NrfvoqSzTPMxbo")

    wine['quality'] = wine['quality'].map_partitions(pd.cut, bins=bins, labels=group_names)

    # Now lets assign a labels to our quality variable
    label_quality = LabelEncoder()
    # Bad becomes 0 and good becomes 1
    wine["quality"] = label_quality.fit_transform(wine["quality"])

    wine.to_csv("/opt/airflow/data/wine.csv", index=False, single_file=True)

    context['ti'].xcom_push(key='wine_data', value="/opt/airflow/data/wine.csv")


@task(task_id=f"transform_task")
def transform_and_load(**context):
    wine_path = context.get("ti").xcom_pull(key="wine_data")
    wine = read_csv(wine_path)

    # Now seperate the dataset as response variable and feature variabes
    X = wine.drop("quality", axis=1)
    y = wine["quality"]

    # Train and Test splitting of data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    # Applying Standard scaling to get optimized result
    sc = StandardScaler()

    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)

    x_train["target"] = y_train.values
    x_test["target"] = y_test.values

    x_train.to_csv("/opt/airflow/data/x_train.csv", index=False, single_file=True)
    x_test.to_csv("/opt/airflow/data/x_test.csv", index=False, single_file=True)

    context['ti'].xcom_push(key='x_train_data', value="/opt/airflow/data/x_train.csv")
    context['ti'].xcom_push(key='x_test_data', value="/opt/airflow/data/x_test.csv")
    etl_name = f"model_v{datetime.now().year}.{datetime.now().month}.{datetime.now().day}.{datetime.now().hour}"
    context['ti'].xcom_push(key='etl_name', value=etl_name)


with DAG(
    dag_id='etl',
    description="etl pipeline",
    schedule_interval='*/5 * * * *',
    default_args=default_args,
    tags=['etl'],
    params={'stage': 'etl'},
) as dag:

    extract_task = extract()
    transform_task = transform_and_load()

    extract_task >> transform_task

if __name__ == "__main__":
    dag.cli()
