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
model_key = "Wine_models/latest_model.model"  # Replace with dynamic logic if needed

# Local temporary paths
local_model_path = "/tmp/latest_model.model"
local_validation_path = "/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/data/ValidationDataset.csv"  # Same as the training file directory

# Step 3: Load Validation Dataset
data = spark.read.csv(local_validation_path, header=True, inferSchema=True, sep=";")

# Clean column names to remove extra quotes
data = data.toDF(*[col.strip().replace('"', '') for col in data.columns])

# Step 4: Assemble Features
feature_cols = [col for col in data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
validation_data = assembler.transform(data).select("features", "quality")

# Step 5: Download and Load the Best Model
s3.download_file(Bucket=bucket_name, Key=model_key, Filename=local_model_path)
model = RandomForestClassificationModel.load(local_model_path)

# Step 6: Make Predictions
predictions = model.transform(validation_data)

# Step 7: Save Predictions Locally and Upload to S3
output_path_local = "/tmp/WinePredictions.csv"
output_key_s3 = "Wine_models/WinePredictions.csv"

predictions.select("features", "quality", "prediction").write.csv(output_path_local, header=True, mode="overwrite")
s3.upload_file(Filename=output_path_local, Bucket=bucket_name, Key=output_key_s3)

print(f"Predictions saved to S3: s3://{bucket_name}/{output_key_s3}")

# Stop Spark Session
spark.stop()
