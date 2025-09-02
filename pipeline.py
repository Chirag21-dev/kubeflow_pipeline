from kfp import dsl
from kfp.dsl import Dataset, Model, Output, Input
import kfp

# ------------------------------
# Step 1: Load data from Snowflake
# ------------------------------
@dsl.component(
    base_image="python:3.9",
    packages_to_install=["snowflake-connector-python", "pandas", "pyarrow"]
)
def load_data_from_snowflake(output_data: Output[Dataset]):
    import pandas as pd
    import snowflake.connector

    # Connect to Snowflake (use secrets / env vars in production)
    conn = snowflake.connector.connect(
        user="YOUR_USER",
        password="YOUR_PASSWORD",
        account="YOUR_ACCOUNT"
    )

    query = "SELECT * FROM YOUR_DB.YOUR_SCHEMA.YOUR_TABLE LIMIT 1000"
    df = pd.read_sql(query, conn)

    # Save dataset to output path
    df.to_parquet(output_data.path)

# ------------------------------
# Step 2: Train model
# ------------------------------
@dsl.component(
    base_image="python:3.9",
    packages_to_install=["scikit-learn", "pandas", "joblib"]
)
def train_model(input_data: Input[Dataset], output_model: Output[Model]):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import joblib

    # Load dataset
    df = pd.read_parquet(input_data.path)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, output_model.path)

# ------------------------------
# Step 3: Save model to GCS/S3
# ------------------------------
@dsl.component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-storage", "boto3"]
)
def save_model_to_bucket(model: Input[Model], bucket_uri: str):
    import os
    from google.cloud import storage
    import boto3
    from urllib.parse import urlparse

    uri = urlparse(bucket_uri)
    if uri.scheme == "gs":
        client = storage.Client()
        bucket = client.bucket(uri.netloc)
        blob = bucket.blob(os.path.join(uri.path.strip("/"), "model.joblib"))
        blob.upload_from_filename(model.path)
    elif uri.scheme == "s3":
        s3 = boto3.client("s3")
        s3.upload_file(model.path, uri.netloc, uri.path.strip("/") + "/model.joblib")
    else:
        raise ValueError("Only gs:// or s3:// supported")

# ------------------------------
# Pipeline definition
# ------------------------------
@dsl.pipeline(
    name="snowflake-train-store-pipeline",
    description="Load data from Snowflake, train ML model, save to bucket"
)
def pipeline(bucket_uri: str = "gs://your-bucket/ml-models/"):
    data = load_data_from_snowflake()
    model = train_model(input_data=data.outputs["output_data"])
    save_model_to_bucket(model=model.outputs["output_model"], bucket_uri=bucket_uri)

# ------------------------------
# Compile pipeline
# ------------------------------
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="snowflake_train_store_pipeline.yaml"
    )
