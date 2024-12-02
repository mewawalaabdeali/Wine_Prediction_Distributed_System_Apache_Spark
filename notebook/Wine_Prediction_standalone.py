import boto3
import os
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
import shutil

# Step 1: Initialize Spark Session
spark = SparkSession.builder \
    .appName("Wine_Quality_Prediction") \
    .master("local[*]") \
    .getOrCreate()

# Step 2: S3 Configurations
s3 = boto3.client('s3')
bucket_name = "winepredictionabdeali"
model_key = "Wine_models/RandomForest_model"  # Ensure this points to the correct model directory

# Local paths
local_model_dir = "/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/tmp/model"
local_validation_path = "/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/data/ValidationDataset.csv"

# Step 3: Load Validation Dataset
data = spark.read.csv(local_validation_path, header=True, inferSchema=True, sep=";")

# Clean column names to remove extra quotes
data = data.toDF(*[col.strip().replace('"', '') for col in data.columns])

# Step 4: Assemble Features
feature_cols = [col for col in data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
validation_data = assembler.transform(data).select("features", "quality")

# Step 5: Download the Best Model from S3
if os.path.exists(local_model_dir):
    shutil.rmtree(local_model_dir)  # Clear the local directory if it exists

os.makedirs(local_model_dir, exist_ok=True)

# Download files directly into the correct local_model_dir
for obj in s3.list_objects_v2(Bucket=bucket_name, Prefix=model_key)["Contents"]:
    key = obj["Key"]
    if key.endswith("/"):  # Skip folders
        continue
    filename = os.path.basename(key)  # Extract the filename
    local_file_path = os.path.join(local_model_dir, filename)  # Place directly into local_model_dir
    s3.download_file(bucket_name, key, local_file_path)

# Load the model
model = RandomForestClassificationModel.load(local_model_dir)

# Step 6: Make Predictions
predictions = model.transform(validation_data)

# Step 7: Save Predictions Locally and Upload to S3
output_path_local = "/tmp/WinePredictions.csv"
output_key_s3 = "Wine_models/WinePredictions.csv"

# Save predictions as a single CSV
predictions.select("features", "quality", "prediction").write.csv(
    path=output_path_local, header=True, mode="overwrite")

# Upload the consolidated predictions to S3
s3.upload_file(output_path_local, bucket_name, output_key_s3)

print(f"Predictions saved to S3: s3://{bucket_name}/{output_key_s3}")

# Stop Spark Session
spark.stop()
