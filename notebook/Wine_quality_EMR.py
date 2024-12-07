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

# Debugging Utility Function
def debug_print(message):
    try:
        print(message)
    except Exception as e:
        pass  # Avoid breaking execution on print failure


debug_print("Starting the script...")

# Step 1: Initialize Spark Session for Cluster Mode
from pyspark import SparkConf

try:
    conf = SparkConf().setAppName("Wine_Quality_Training_Distributed")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    debug_print("Spark session initialized in cluster mode.")
except Exception as e:
    debug_print(f"Error initializing Spark session: {e}")
    raise e

# Step 2: Load and Clean Data
try:
    data_path = "s3://winepredictionabdeali/TrainingDataset.csv"  # S3 path for training data
    data = spark.read.csv(data_path, header=True, inferSchema=True, sep=";")
    data = data.toDF(*[col.strip().replace('"', '') for col in data.columns])  # Clean column names
    debug_print(f"Data loaded from {data_path} with {data.count()} rows and {len(data.columns)} columns.")
except Exception as e:
    debug_print(f"Error loading data: {e}")
    raise e

# Step 3: Train-Test Split
try:
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    debug_print("Data split into training and test sets.")
except Exception as e:
    debug_print(f"Error splitting data: {e}")
    raise e

# Step 4: Feature Engineering
try:
    feature_cols = [col for col in train_data.columns if col != "quality"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)
    debug_print("Feature engineering setup complete.")
except Exception as e:
    debug_print(f"Error in feature engineering: {e}")
    raise e

# Step 5: Model and Hyperparameter Tuning
try:
    rf = RandomForestClassifier(labelCol="quality", featuresCol="scaled_features", seed=42)
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 50]) \
        .addGrid(rf.maxDepth, [5, 10]) \
        .build()

    crossval = CrossValidator(estimator=rf,
                              estimatorParamMaps=paramGrid,
                              evaluator=MulticlassClassificationEvaluator(labelCol="quality", metricName="accuracy"),
                              numFolds=3)

    pipeline = Pipeline(stages=[assembler, scaler, crossval])
    debug_print("Pipeline and hyperparameter tuning setup complete.")
except Exception as e:
    debug_print(f"Error in model setup: {e}")
    raise e

# Step 6: Train Model
try:
    debug_print("Starting model training with hyperparameter tuning...")
    pipeline_model = pipeline.fit(train_data)
    debug_print("Model training completed.")
except Exception as e:
    debug_print(f"Error during model training: {e}")
    raise e

# Step 7: Evaluate Model
try:
    debug_print("Starting model evaluation...")
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")
    train_prediction = pipeline_model.transform(train_data)
    test_prediction = pipeline_model.transform(test_data)
    train_f1 = evaluator.evaluate(train_prediction, {evaluator.metricName: "f1"})
    test_f1 = evaluator.evaluate(test_prediction, {evaluator.metricName: "f1"})
    train_accuracy = evaluator.evaluate(train_prediction, {evaluator.metricName: "accuracy"})
    test_accuracy = evaluator.evaluate(test_prediction, {evaluator.metricName: "accuracy"})

    # Enhanced Metrics Output
    debug_print("\n")
    debug_print("***********************************************************************")
    debug_print("+++++++++++++++++++++++ Model Evaluation Metrics ++++++++++++++++++++++")
    debug_print("***********************************************************************")
    debug_print(f"[Train] F1 Score      : {train_f1:.4f}")
    debug_print(f"[Train] Accuracy      : {train_accuracy:.4f}")
    debug_print("-----------------------------------------------------------------------")
    debug_print(f"[Test]  F1 Score      : {test_f1:.4f}")
    debug_print(f"[Test]  Accuracy      : {test_accuracy:.4f}")
    debug_print("***********************************************************************")
except Exception as e:
    debug_print(f"Error during evaluation: {e}")
    raise e

# Step 8: Save Model to S3
try:
    # Generate unique hash for the model
    param_str = json.dumps({"numTrees": [10, 50], "maxDepth": [5, 10]}, sort_keys=True)
    model_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    model_name = f"RandomForest_Acc{int(test_accuracy*100)}_{model_hash}"
    local_model_dir = f"/tmp/{model_name}"
    shutil.rmtree(local_model_dir, ignore_errors=True)
    pipeline_model.write().save(local_model_dir)
    debug_print(f"Model successfully saved locally at: {local_model_dir}")

    # Upload to S3
    s3_client = boto3.client('s3')
    bucket_name = "winepredictionabdeali"
    s3_path = f"Testing_models/{model_name}"

    debug_print(f"Uploading model files to S3 path: s3://{bucket_name}/{s3_path}/")
    for root, dirs, files in os.walk(local_model_dir):
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, local_model_dir)
            s3_key = f"{s3_path}/{relative_path}"
            s3_client.upload_file(full_path, bucket_name, s3_key)
    debug_print(f"Model successfully uploaded to S3: s3://{bucket_name}/{s3_path}/")
except Exception as e:
    debug_print(f"Error during model save or upload: {e}")
    raise e

# Step 9: Stop Spark Session
try:
    spark.stop()
    debug_print("Spark session stopped.")
except Exception as e:
    debug_print(f"Error stopping Spark session: {e}")
    raise e
