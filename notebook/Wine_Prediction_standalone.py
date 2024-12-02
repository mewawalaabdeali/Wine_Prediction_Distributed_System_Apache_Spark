import boto3
import os
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler

# Step 1: Initialize Spark Session
spark = SparkSession.builder \
    .appName("Wine_Quality_Prediction") \
    .master("local[*]") \
    .getOrCreate()

# Step 2: S3 Configurations
s3 = boto3.client('s3')
bucket_name = "winepredictionabdeali"
s3_model_prefix = "Wine_models/RandomForest_model"  # S3 prefix for the model directory

# Local paths
local_model_dir = "/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark"
local_validation_path = "/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/data/ValidationDataset.csv"

# Step 3: Load Validation Dataset
data = spark.read.csv(local_validation_path, header=True, inferSchema=True, sep=";")

# Clean column names to remove extra quotes
data = data.toDF(*[col.strip().replace('"', '') for col in data.columns])

# Step 4: Assemble Features
feature_cols = [col for col in data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
validation_data = assembler.transform(data).select("features", "quality")

# Step 5: Download Model Files from S3
# Ensure the local model directory exists
os.makedirs(local_model_dir, exist_ok=True)

# Download model files directly into the local_model_dir
print(f"Downloading model files from S3: s3://{bucket_name}/{s3_model_prefix}")
paginator = s3.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_model_prefix):
    if "Contents" in page:
        for obj in page["Contents"]:
            s3_file_key = obj["Key"]
            file_name = os.path.basename(s3_file_key)  # Extract file name only
            local_file_path = os.path.join(local_model_dir, file_name)  # Flat structure
            s3.download_file(bucket_name, s3_file_key, local_file_path)
            print(f"Downloaded {s3_file_key} to {local_file_path}")

# Step 6: Load the Model
try:
    model = RandomForestClassificationModel.load(local_model_dir)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Step 7: Make Predictions
predictions = model.transform(validation_data)

# Step 8: Save Predictions Locally and Upload to S3
output_path_local = "/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/WinePredictions.csv"
output_key_s3 = "Wine_models/WinePredictions.csv"

# Save predictions as a single CSV locally
predictions.select("features", "quality", "prediction").write.csv(
    path=output_path_local, header=True, mode="overwrite"
)

# Upload the predictions file to S3
s3.upload_file(output_path_local, bucket_name, output_key_s3)
print(f"Predictions uploaded to s3://{bucket_name}/{output_key_s3}")
