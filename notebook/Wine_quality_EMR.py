import boto3
import os
from urllib.parse import urlparse
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import json
import hashlib

# Step 1: Initialize Spark Session for Cluster Mode
from pyspark import SparkConf

conf = SparkConf().setAppName("Wine_Quality_Training_Distributed")
spark = SparkSession.builder.config(conf=conf).getOrCreate()
print("Spark session initialized in cluster mode.")

# Step 2: Load and Clean Data
data_path = "s3://winepredictionabdeali/TrainingDataset.csv"
data = spark.read.csv(data_path, header=True, inferSchema=True, sep=";")
data = data.toDF(*[col.strip().replace('"', '') for col in data.columns])
print(f"Data loaded from {data_path} with {data.count()} rows and {len(data.columns)} columns.")

# Step 3: Train-Test Split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Step 4: Feature Engineering
feature_cols = [col for col in train_data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)

# Step 5: Model and Hyperparameter Tuning
rf = RandomForestClassifier(labelCol="quality", featuresCol="scaled_features", seed=42)
paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [10, 50]).addGrid(rf.maxDepth, [5, 10]).build()
crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(labelCol="quality", metricName="accuracy"),
                          numFolds=3)

# Step 6: Create Pipeline
pipeline = Pipeline(stages=[assembler, scaler, crossval])

# Step 7: Train Model
pipeline_model = pipeline.fit(train_data)
print("Model training completed.")

# Step 8: Model Save Logic
s3_bucket = "winepredictionabdeali"
s3_model_path = "Testing_models"
model_name = f"RandomForest_{hashlib.md5(json.dumps(paramGrid).encode()).hexdigest()[:8]}"
local_model_dir = f"/tmp/{model_name}"

# Delete Existing S3 Directory if it Exists
def s3_delete_and_overwrite(bucket_name, folder_name):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    bucket.objects.filter(Prefix=folder_name).delete()
    print(f"Deleted pre-existing folder in S3: {folder_name}")

# Check for Folder Existence
def folder_exists(bucket_name, folder_name):
    s3 = boto3.client('s3')
    res = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
    return 'Contents' in res

# Implement Folder Deletion
if folder_exists(s3_bucket, f"{s3_model_path}/{model_name}/"):
    s3_delete_and_overwrite(s3_bucket, f"{s3_model_path}/{model_name}/")

# Save Model Locally
pipeline_model.write().overwrite().save(local_model_dir)
print(f"Model saved locally at: {local_model_dir}")

# Upload Model to S3
s3_client = boto3.client('s3')
for root, dirs, files in os.walk(local_model_dir):
    for file in files:
        full_path = os.path.join(root, file)
        relative_path = os.path.relpath(full_path, local_model_dir)
        s3_key = os.path.join(s3_model_path, model_name, relative_path).replace("\\", "/")
        s3_client.upload_file(full_path, s3_bucket, s3_key)
        print(f"Uploaded {full_path} to s3://{s3_bucket}/{s3_key}")

# Step 9: Evaluate Model
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")
train_prediction = pipeline_model.transform(train_data)
test_prediction = pipeline_model.transform(test_data)
train_accuracy = evaluator.evaluate(train_prediction, {evaluator.metricName: "accuracy"})
test_accuracy = evaluator.evaluate(test_prediction, {evaluator.metricName: "accuracy"})

print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Step 10: Stop Spark Session
spark.stop()
print("Spark session stopped.")
