import boto3
import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from io import StringIO
import pandas as pd

# Step 1: Capture Command-Line Arguments
if len(sys.argv) != 3:
    print("Usage: python prediction.py <validation_file_path_or_s3> <model_folder_name_or_s3>")
    sys.exit(1)

validation_data_path = sys.argv[1]  # Validation dataset path (local or S3)
model_path = sys.argv[2]  # Model folder path (local or S3)

# Step 2: Initialize Spark Session
spark = SparkSession.builder \
    .appName("Wine_Quality_Prediction") \
    .master("local[*]") \
    .getOrCreate()

# Step 3: Initialize S3 Client
s3_client = boto3.client('s3')

# Step 4: Handle Model Path
local_model_dir = "/app/models"  # Base directory for model storage in the container
os.makedirs(local_model_dir, exist_ok=True)

if model_path.startswith("s3://"):
    print(f"Downloading model from S3: {model_path}")
    s3_bucket, s3_prefix = model_path[5:].split("/", 1)
    local_model_dir = os.path.join(local_model_dir, os.path.basename(s3_prefix))

    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
    for obj in response.get('Contents', []):
        s3_key = obj['Key']
        local_path = os.path.join(local_model_dir, os.path.relpath(s3_key, s3_prefix))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(s3_bucket, s3_key, local_path)

    print(f"Model downloaded to: {local_model_dir}")
else:
    local_model_dir = os.path.join("/app", model_path)
    print(f"Using local model path inside container: {local_model_dir}")

# Verify metadata exists in model directory
metadata_path = os.path.join(local_model_dir, "metadata")
if not os.path.exists(metadata_path):
    print(f"Error: Metadata file not found in model directory: {metadata_path}")
    sys.exit(1)

# Step 5: Load Model
pipeline_model = PipelineModel.load(local_model_dir)
print(f"Model loaded successfully from: {local_model_dir}")

# Step 6: Handle Validation File
local_validation_path = "/app/data/ValidationDataset.csv"  # Default path for validation dataset in the container

if validation_data_path.startswith("s3://"):
    s3_validation_bucket, s3_validation_key = validation_data_path[5:].split("/", 1)
    s3_client.download_file(s3_validation_bucket, s3_validation_key, local_validation_path)
    print(f"Validation file downloaded from S3: {validation_data_path}")
else:
    local_validation_path = os.path.join("/app", validation_data_path)
    print(f"Using local validation file inside container: {local_validation_path}")

# Load validation dataset
validation_data = spark.read.csv(local_validation_path, header=True, inferSchema=True, sep=";")
validation_data = validation_data.toDF(*[col.strip().replace('"', '') for col in validation_data.columns])

# Step 7: Make Predictions
predictions = pipeline_model.transform(validation_data)

# Print predictions to console
print("\nPredictions:")
predictions.select("quality", "prediction").show(truncate=False)

# Step 8: Evaluate Model
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")
metrics = {
    "Accuracy": evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"}),
    "Precision": evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"}),
    "Recall": evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"}),
    "F1 Score": evaluator.evaluate(predictions, {evaluator.metricName: "f1"}),
}

print("\nPrediction Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Step 9: Upload Predictions Directly to S3
predictions_df = predictions.select("quality", "prediction").toPandas()

# Save predictions to S3
predictions_s3_key = "Wine_models/WinePredictions.csv"
csv_buffer = StringIO()
predictions_df.to_csv(csv_buffer, index=False)
s3_client.put_object(Bucket="winepredictionabdeali", Key=predictions_s3_key, Body=csv_buffer.getvalue())
print(f"Predictions uploaded directly to S3: s3://winepredictionabdeali/{predictions_s3_key}")

# Stop Spark Session
spark.stop()
print("Spark session stopped.")
