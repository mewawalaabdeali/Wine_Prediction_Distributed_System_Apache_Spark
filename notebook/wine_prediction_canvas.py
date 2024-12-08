import boto3
import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from io import StringIO
import pandas as pd
from pyspark.sql.functions import col

# Step 1: Capture Command-Line Arguments
if len(sys.argv) != 3:
    print("Usage: spark-submit wine_prediction_canvas.py <validation_file_path_or_s3> <model_folder_name>")
    sys.exit(1)

validation_data_path = sys.argv[1]  # Validation dataset path (local or S3)
model_folder_name = sys.argv[2]  # Folder name of the saved model (e.g., "PipelineModel_20241203152045")

# Step 2: Initialize Spark Session
spark = SparkSession.builder \
    .appName("Wine_Quality_Prediction") \
    .getOrCreate()  # Automatically determines whether to run in local or cluster mode

print("Spark session initialized.")

# Step 3: S3 Configuration
s3_client = boto3.client('s3')
bucket_name = "winepredictionabdealicanvas"  # The bucket to store the models
model_dir = f"/home/hadoop/Wine_Prediction_Distributed_System_Apache_Spark/models/{model_folder_name}"  # Local model directory

# Step 4: Download Model from S3 (if not already downloaded)
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

# Step 5: Load Model (Ensure the path is correct for local mode)
pipeline_model = PipelineModel.load(model_dir)  # Ensure this points to the correct local model directory
print(f"Model loaded successfully from: {model_dir}")

# Step 6: Handle Validation File
if validation_data_path.startswith("s3://"):
    # If validation file is in S3, download it
    s3_validation_bucket, s3_validation_key = validation_data_path[5:].split("/", 1)
    local_validation_path = "/home/hadoop/Wine_Prediction_Distributed_System_Apache_Spark/data/ValidationDataset.csv"
    s3_client.download_file(s3_validation_bucket, s3_validation_key, local_validation_path)
    print(f"Validation file downloaded from S3: {validation_data_path}")
else:
    # Assume local file path
    local_validation_path = validation_data_path
    print(f"Using local validation file: {local_validation_path}")

# Step 7: Load Validation Dataset
validation_data = spark.read.csv(local_validation_path, header=True, inferSchema=True, sep=";")
validation_data = validation_data.toDF(*[col.strip().replace('"', '') for col in validation_data.columns])

# Step 8: Make Predictions
predictions = pipeline_model.transform(validation_data)

# Print predictions to console
print("\nPredictions:")
predictions.select("quality", "prediction").show(truncate=False)

# Step 9: Evaluate Model
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

# Step 10: Upload Predictions Directly to S3
predictions_df = predictions.select("quality", "prediction").toPandas()

# Save predictions to S3
predictions_s3_key = "Wine_models/WinePredictions.csv"
csv_buffer = StringIO()
predictions_df.to_csv(csv_buffer, index=False)
s3_client.put_object(Bucket=bucket_name, Key=predictions_s3_key, Body=csv_buffer.getvalue())
print(f"Predictions uploaded directly to S3: s3://{bucket_name}/{predictions_s3_key}")

# Step 11: Stop Spark Session
spark.stop()
print("Spark session stopped.")
