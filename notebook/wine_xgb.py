import boto3 
import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from datetime import datetime
from pyspark.sql.functions import col, when
from pyspark import SparkConf
import numpy as np

# Step 1: Initialize Spark Session for Cluster Mode
conf = SparkConf().setAppName("Wine_Quality_Training_Distributed")
spark = SparkSession.builder.config(conf=conf).getOrCreate()

print("Spark session initialized in cluster mode.")

# Step 2: Load and Clean Data
data_path = "s3://winepredictionabdeali/TrainingDataset.csv"  # S3 path for training data
data = spark.read.csv(data_path, header=True, inferSchema=True, sep=";")
data = data.toDF(*[col.strip().replace('"', '') for col in data.columns])  # Clean column names
print(f"Data loaded from {data_path} with {data.count()} rows and {len(data.columns)} columns.")

# Step 3: Handle Class Imbalance with Sampling (Random OverSampling)
# Count of each class in the 'quality' column
class_counts = data.groupBy('quality').count().collect()
class_counts = {row['quality']: row['count'] for row in class_counts}

# Get the majority class count
max_class_count = max(class_counts.values())

# Create an empty DataFrame for oversampling
oversampled_df = None

# For each class, generate synthetic data (oversampling)
for quality, count in class_counts.items():
    # Filter the data for this specific class
    class_data = data.filter(col("quality") == quality)
    
    # Calculate how many samples are needed for oversampling
    needed_samples = max_class_count - count
    
    # Create a DataFrame with duplicates for oversampling
    oversampled_class_data = class_data.unionAll(class_data.limit(needed_samples))
    
    # Append oversampled class data to the final DataFrame
    if oversampled_df is None:
        oversampled_df = oversampled_class_data
    else:
        oversampled_df = oversampled_df.unionAll(oversampled_class_data)

# Now oversampled_df contains the balanced data
print(f"Balanced dataset size: {oversampled_df.count()}")

# Step 4: Train-Test Split
train_data, test_data = oversampled_df.randomSplit([0.8, 0.2], seed=42)

# Step 5: Feature Engineering
feature_cols = [col for col in train_data.columns if col != "quality"]

# Assemble the features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)

# Step 6: Model and Hyperparameter Tuning (Optimized)
# RandomForest Classifier
rf = RandomForestClassifier(labelCol="quality", featuresCol="scaled_features", seed=42)

# DecisionTree Classifier
dt = DecisionTreeClassifier(labelCol="quality", featuresCol="scaled_features", seed=42)

# Parameter Grid for RandomForest
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 50]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()

# Cross-validation for RandomForest
crossval_rf = CrossValidator(estimator=rf,
                              estimatorParamMaps=paramGrid,
                              evaluator=MulticlassClassificationEvaluator(labelCol="quality", metricName="accuracy"),
                              numFolds=3)

# Cross-validation for DecisionTree
crossval_dt = CrossValidator(estimator=dt,
                              estimatorParamMaps=paramGrid,
                              evaluator=MulticlassClassificationEvaluator(labelCol="quality", metricName="accuracy"),
                              numFolds=3)

# Step 7: Create Pipeline
pipeline_rf = Pipeline(stages=[assembler, scaler, crossval_rf])
pipeline_dt = Pipeline(stages=[assembler, scaler, crossval_dt])

# Step 8: Train Model (RandomForest)
print("Training RandomForest model with hyperparameter tuning...")
pipeline_model_rf = pipeline_rf.fit(train_data)
print("RandomForest model training completed.")

# Step 9: Train Model (DecisionTree)
print("Training DecisionTree model with hyperparameter tuning...")
pipeline_model_dt = pipeline_dt.fit(train_data)
print("DecisionTree model training completed.")

# Step 10: Evaluate Models
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
accuracy_rf = evaluator.evaluate(pipeline_model_rf.transform(test_data))
accuracy_dt = evaluator.evaluate(pipeline_model_dt.transform(test_data))

# Step 11: Save the Best Model to S3
best_model = None
model_name = ""
if accuracy_rf > accuracy_dt:
    best_model = pipeline_model_rf
    model_name = "RandomForestModel"
    print(f"RandomForest model performs better with accuracy: {accuracy_rf}")
else:
    best_model = pipeline_model_dt
    model_name = "DecisionTreeModel"
    print(f"DecisionTree model performs better with accuracy: {accuracy_dt}")

# Step 12: Save Best Model to S3
s3_client = boto3.client('s3')
bucket_name = "winepredictionabdealicanvas"  # Updated S3 bucket
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
model_dir = f"s3a://winepredictionabdeali/Testing_models/{model_name}_{timestamp}"

best_model.write().overwrite().save(model_dir)
print(f"Best model saved to: {model_dir}")

# Step 13: Print Model Name for Jenkins
print(f"MODEL_NAME={model_name}_{timestamp}")

# Step 14: Evaluate Model Metrics
train_prediction = best_model.transform(train_data)
test_prediction = best_model.transform(test_data)

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

# Step 15: Stop Spark Session
spark.stop()
print("Spark session stopped.")
