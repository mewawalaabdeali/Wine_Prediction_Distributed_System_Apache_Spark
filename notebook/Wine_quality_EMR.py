from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import hashlib
import os
import shutil
import boto3
import json

# Step 1: Initialize Spark Session for Cluster Mode
from pyspark import SparkConf

conf = SparkConf().setAppName("Wine_Quality_Training_Distributed")
spark = SparkSession.builder.config(conf=conf).getOrCreate()

print("Spark session initialized in cluster mode.")

# Step 2: Load and Clean Data
data_path = "s3://winepredictionabdeali/TrainingDataset.csv"  # S3 path for training data
data = spark.read.csv(data_path, header=True, inferSchema=True, sep=";")
data = data.toDF(*[col.strip().replace('"', '') for col in data.columns])  # Clean column names
print(f"Data loaded from {data_path} with {data.count()} rows and {len(data.columns)} columns.")

# Step 3: Train-Test Split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Step 4: Feature Engineering
feature_cols = [col for col in train_data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)

# Step 5: Model and Hyperparameter Tuning (Optimized)
rf = RandomForestClassifier(labelCol="quality", featuresCol="scaled_features", seed=42)

# Optimized parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 50]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()

# Reduced cross-validation folds for faster processing
crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(labelCol="quality", metricName="accuracy"),
                          numFolds=3)  # Reduced folds

# Step 6: Create Pipeline
pipeline = Pipeline(stages=[assembler, scaler, crossval])

# Step 7: Train Model
print("Starting model training with hyperparameter tuning...")
pipeline_model = pipeline.fit(train_data)
print("Model training completed with hyperparameter tuning.")

# Step 8: Evaluate Model
print("Starting model evaluation...")
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")
test_prediction = pipeline_model.transform(test_data)

# Calculate Evaluation Metrics
test_accuracy = evaluator.evaluate(test_prediction, {evaluator.metricName: "accuracy"})
print(f"Test Accuracy: {test_accuracy:.4f}")

# Step 9: Save Model to S3 with Unique Naming
# Generate a unique hash for the model based on parameters and test accuracy
param_str = json.dumps({"numTrees": [10, 50], "maxDepth": [5, 10]}, sort_keys=True)
model_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]  # Generate a short hash
model_name = f"RandomForest_Acc{int(test_accuracy*100)}_{model_hash}"  # Use test accuracy and hash
local_model_dir = f"/tmp/{model_name}"  # Temporary local storage for model

# Remove local directory if it exists to avoid conflicts
if os.path.exists(local_model_dir):
    shutil.rmtree(local_model_dir)

# Save the model locally
pipeline_model.write().save(local_model_dir)
print(f"Model saved locally at: {local_model_dir}")

# Upload the model directory to S3 without overwriting
s3_client = boto3.client('s3')
bucket_name = "winepredictionabdeali"
s3_path = f"Testing_models/{model_name}"  # Updated S3 directory
for root, dirs, files in os.walk(local_model_dir):
    for file in files:
        full_path = os.path.join(root, file)
        s3_key = os.path.relpath(full_path, local_model_dir)
        s3_client.upload_file(full_path, bucket_name, f"{s3_path}/{s3_key}")
print(f"Model uploaded to S3: s3://{bucket_name}/{s3_path}/")

# Print the model name for Jenkins
print(f"MODEL_NAME={model_name}")

# Step 10: Stop Spark Session
spark.stop()
print("Spark session stopped.")
