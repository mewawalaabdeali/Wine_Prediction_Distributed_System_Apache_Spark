import boto3
import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# Step 1: Capture Command-Line Arguments
if len(sys.argv) != 2:
    print("Usage: python prediction.py <validation_file_path>")
    sys.exit(1)

validation_data_path = sys.argv[1]  # Validation dataset path

# Step 2: Initialize Spark Session
spark = SparkSession.builder \
    .appName("Wine_Quality_Prediction") \
    .master("local[*]") \
    .getOrCreate()

# Step 3: S3 Configuration
s3_client = boto3.client('s3')
bucket_name = "winepredictionabdeali"
model_dir = "/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/models/PipelineModel"

# Step 4: Download Model from S3
os.makedirs(model_dir, exist_ok=True)
s3_prefix = "Wine_models"
for obj in s3_client.list_objects(Bucket=bucket_name, Prefix=s3_prefix).get('Contents', []):
    s3_key = obj['Key']
    local_path = os.path.join(model_dir, os.path.basename(s3_key))
    s3_client.download_file(bucket_name, s3_key, local_path)
print(f"Model downloaded from S3 to: {model_dir}")

# Step 5: Load Model and Validation Dataset
pipeline_model = PipelineModel.load(model_dir)
validation_data = spark.read.csv(validation_data_path, header=True, inferSchema=True, sep=";")
validation_data = validation_data.toDF(*[col.strip().replace('"', '') for col in validation_data.columns])

# Step 6: Make Predictions
predictions = pipeline_model.transform(validation_data)
predictions.select("features", "quality", "prediction").show()

# Stop Spark Session
spark.stop()
