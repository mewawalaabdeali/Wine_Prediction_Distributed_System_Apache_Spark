import findspark
findspark.init()  # Initialize Spark to ensure paths are correctly set

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from datetime import datetime
import os
import shutil
import boto3

# Step 1: Initialize Spark Session for Cluster Mode
#spark = SparkSession.builder \
#    .appName("Wine_Quality_Training_Distributed") \
#   .config("spark.jars.packages", "org.apache.spark:spark-hadoop-cloud_2.12:3.3.0") \
#    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
#    .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
#    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
#    .getOrCreate()

spark = SparkSession.builder \
    .appName("WineQualityPrediction") \
    .config("spark.driver.memory", "2").getOrCreate()


print("Spark session initialized in cluster mode.")

# Step 2: Load and Clean Data
data_path = "s3a://winepredictionabdeali/TrainingDataset.csv"  # Changed to s3a:// to match Hadoop configuration
data = spark.read.csv(data_path, header=True, inferSchema=True, sep=";")
data = data.toDF(*[col.strip().replace('"', '') for col in data.columns])  # Clean column names
print(f"Data loaded from {data_path} with {data.count()} rows and {len(data.columns)} columns.")

# Step 3: Train-Test Split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Step 4: Feature Engineering
feature_cols = [col for col in train_data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)

# Step 5: Model and Hyperparameter Tuning
rf = RandomForestClassifier(labelCol="quality", featuresCol="scaled_features", seed=42)
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 50, 100]) \
    .addGrid(rf.maxDepth, [5, 10, 20]) \
    .build()

crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(labelCol="quality", metricName="accuracy"),
                          numFolds=5)  # 5-fold cross-validation

# Step 6: Create Pipeline
pipeline = Pipeline(stages=[assembler, scaler, crossval])

# Step 7: Train Model
pipeline_model = pipeline.fit(train_data)
print("Model training completed with hyperparameter tuning.")

# Step 8: Save Model to S3
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
model_name = f"PipelineModel_{timestamp}"
local_model_dir = f"/tmp/{model_name}"  # Temporary local storage for model
shutil.rmtree(local_model_dir, ignore_errors=True)  # Clear existing directory
pipeline_model.write().overwrite().save(local_model_dir)
print(f"Model saved locally at: {local_model_dir}")

# Upload the model directory to S3
s3_client = boto3.client('s3')
bucket_name = "winepredictionabdeali"
for root, dirs, files in os.walk(local_model_dir):
    for file in files:
        full_path = os.path.join(root, file)
        s3_key = os.path.relpath(full_path, local_model_dir)
        s3_client.upload_file(full_path, bucket_name, f"Wine_models/{model_name}/{s3_key}")
print(f"Model uploaded to S3: s3://{bucket_name}/Wine_models/{model_name}/")

# Print the model name for Jenkins
print(f"MODEL_NAME={model_name}")

# Step 9: Evaluate Model
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")
train_prediction = pipeline_model.transform(train_data)
test_prediction = pipeline_model.transform(test_data)

# Calculate Evaluation Metrics
train_f1 = evaluator.evaluate(train_prediction, {evaluator.metricName: "f1"})
test_f1 = evaluator.evaluate(test_prediction, {evaluator.metricName: "f1"})
train_accuracy = evaluator.evaluate(train_prediction, {evaluator.metricName: "accuracy"})
test_accuracy = evaluator.evaluate(test_prediction, {evaluator.metricName: "accuracy"})

# Enhanced Metrics Output
print("\n")
print("***********************************************************************")
print("+++++++++++++++++++++++ Model Evaluation Metrics ++++++++++++++++++++++")
print("***********************************************************************")
print(f"[Train] F1 Score      : {train_f1:.4f}")
print(f"[Train] Accuracy      : {train_accuracy:.4f}")
print("-----------------------------------------------------------------------")
print(f"[Test]  F1 Score      : {test_f1:.4f}")
print(f"[Test]  Accuracy      : {test_accuracy:.4f}")
print("***********************************************************************")

# Step 10: Stop Spark Session
spark.stop()
print("Spark session stopped.")
