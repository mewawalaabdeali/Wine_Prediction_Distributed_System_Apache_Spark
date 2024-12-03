import boto3
import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# Step 1: Capture Command-Line Arguments
if len(sys.argv) != 3:
    print("Usage: python prediction.py <validation_file_path> <model_folder_name>")
    sys.exit(1)

validation_data_path = sys.argv[1]  # Validation dataset path
model_folder_name = sys.argv[2]  # Folder name of the saved model (e.g., "PipelineModel_20241203152045")

# Step 2: Initialize Spark Session
spark = SparkSession.builder \
    .appName("Wine_Quality_Prediction") \
    .master("local[*]") \
    .getOrCreate()

# Step 3: S3 Configuration
s3_client = boto3.client('s3')
bucket_name = "winepredictionabdeali"
model_dir = f"/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/models/{model_folder_name}"

# Step 4: Download Model from S3
if not os.path.exists(model_dir):
    print(f"Downloading model from S3: s3://{bucket_name}/Wine_models/{model_folder_name}/")
    os.makedirs(model_dir, exist_ok=True)
    s3_prefix = f"Wine_models/{model_folder_name}"
    for obj in s3_client.list_objects(Bucket=bucket_name, Prefix=s3_prefix).get('Contents', []):
        s3_key = obj['Key']
        local_path = os.path.join(model_dir, os.path.basename(s3_key))
        s3_client.download_file(bucket_name, s3_key, local_path)
    print(f"Model downloaded to: {model_dir}")
else:
    print(f"Model found locally at: {model_dir}")

# Step 5: Load Model and Validation Dataset
pipeline_model = PipelineModel.load(model_dir)
print(f"Model loaded successfully from: {model_dir}")

validation_data = spark.read.csv(validation_data_path, header=True, inferSchema=True, sep=";")
validation_data = validation_data.toDF(*[col.strip().replace('"', '') for col in validation_data.columns])

# Step 6: Make Predictions
predictions = pipeline_model.transform(validation_data)
predictions.select("features", "quality", "prediction").show()

# Step 7: Save Predictions Locally and Upload to S3
predictions_output_path = "/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/WinePredictions.csv"
predictions.select("features", "quality", "prediction") \
    .write.csv(path=predictions_output_path, header=True, mode="overwrite")

# Upload predictions to S3
predictions_s3_key = "Wine_models/WinePredictions.csv"
s3_client.upload_file(predictions_output_path, bucket_name, predictions_s3_key)
print(f"Predictions uploaded to S3: s3://{bucket_name}/{predictions_s3_key}")

# Stop Spark Session
spark.stop()
print("Spark session stopped.")
