from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
import boto3
import os

# Step 1: Initialize Spark Session
spark = SparkSession.builder \
    .appName("Wine_Quality_Training") \
    .master("local[*]") \
    .getOrCreate()

# Step 2: Load Data
data_path = "/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/data/TrainingDataset.csv"
data = spark.read.csv(data_path, header=True, inferSchema=True, sep=";")
data = data.toDF(*[col.strip().replace('"', '') for col in data.columns])  # Clean column names

# Step 3: Split Data
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Step 4: Create Pipeline
feature_cols = [col for col in train_data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)
rf = RandomForestClassifier(labelCol="quality", featuresCol="scaled_features", seed=42)
pipeline = Pipeline(stages=[assembler, scaler, rf])

# Step 5: Train Model
pipeline_model = pipeline.fit(train_data)

# Step 6: Save Model Locally
model_dir = "/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/models/PipelineModel"
pipeline_model.write().overwrite().save(model_dir)
print(f"Model saved locally at: {model_dir}")

# Step 7: Upload Model to S3
s3_client = boto3.client('s3')
bucket_name = "winepredictionabdeali"
for root, dirs, files in os.walk(model_dir):
    for file in files:
        full_path = os.path.join(root, file)
        s3_key = os.path.relpath(full_path, model_dir)  # Keep flat structure for S3
        s3_client.upload_file(full_path, bucket_name, f"Wine_models/{s3_key}")
print(f"Model uploaded to S3: s3://{bucket_name}/Wine_models/")

# Stop Spark Session
spark.stop()
