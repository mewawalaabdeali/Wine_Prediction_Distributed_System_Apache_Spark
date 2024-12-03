from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import boto3
import os
import shutil
from datetime import datetime

# Step 1: Initialize Spark Session
spark = SparkSession.builder \
    .appName("Wine_Quality_Training") \
    .master("local[*]") \
    .getOrCreate()

print("Spark session initialized successfully.")

# Step 2: Load and Clean Data
data_path = "/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/data/TrainingDataset.csv"
data = spark.read.csv(data_path, header=True, inferSchema=True, sep=";")
data = data.toDF(*[col.strip().replace('"', '') for col in data.columns])  # Clean column names
print(f"Data loaded from {data_path} with {data.count()} rows and {len(data.columns)} columns.")

# Step 3: Split Data
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
print(f"Training set: {train_data.count()} rows, Test set: {test_data.count()} rows.")

# Step 4: Define Pipeline
feature_cols = [col for col in train_data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)
rf = RandomForestClassifier(labelCol="quality", featuresCol="scaled_features", seed=42)
pipeline = Pipeline(stages=[assembler, scaler, rf])

# Step 5: Train Model
pipeline_model = pipeline.fit(train_data)
print("Model training completed.")

# Step 6: Evaluate Model
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")
metrics = {
    "Training Accuracy": evaluator.evaluate(pipeline_model.transform(train_data), {evaluator.metricName: "accuracy"}),
    "Training Precision": evaluator.evaluate(pipeline_model.transform(train_data), {evaluator.metricName: "weightedPrecision"}),
    "Training Recall": evaluator.evaluate(pipeline_model.transform(train_data), {evaluator.metricName: "weightedRecall"}),
    "Training F1 Score": evaluator.evaluate(pipeline_model.transform(train_data), {evaluator.metricName: "f1"}),
    "Test Accuracy": evaluator.evaluate(pipeline_model.transform(test_data), {evaluator.metricName: "accuracy"}),
    "Test Precision": evaluator.evaluate(pipeline_model.transform(test_data), {evaluator.metricName: "weightedPrecision"}),
    "Test Recall": evaluator.evaluate(pipeline_model.transform(test_data), {evaluator.metricName: "weightedRecall"}),
    "Test F1 Score": evaluator.evaluate(pipeline_model.transform(test_data), {evaluator.metricName: "f1"}),
}

print("\nModel Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Step 7: Save Model Locally
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
model_name = f"PipelineModel_{timestamp}"
model_dir = f"/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/models/{model_name}"
shutil.rmtree(model_dir, ignore_errors=True)  # Clear existing directory
pipeline_model.write().overwrite().save(model_dir)
print(f"Model saved locally at: {model_dir}")

# Step 8: Upload Model to S3
s3_client = boto3.client('s3')
bucket_name = "winepredictionabdeali"
for root, dirs, files in os.walk(model_dir):
    for file in files:
        full_path = os.path.join(root, file)
        s3_key = os.path.relpath(full_path, model_dir)
        s3_client.upload_file(full_path, bucket_name, f"Wine_models/{model_name}/{s3_key}")
print(f"Model uploaded to S3: s3://{bucket_name}/Wine_models/{model_name}/")

# Stop Spark Session
spark.stop()
print("Spark session stopped.")
