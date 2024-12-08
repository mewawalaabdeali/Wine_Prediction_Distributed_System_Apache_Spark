import boto3
import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from datetime import datetime
from pyspark.sql.functions import col
from pyspark import SparkConf

# Step 1: Initialize Spark Session for Cluster Mode
conf = SparkConf().setAppName("Wine_Quality_Training_Distributed")
spark = SparkSession.builder.config(conf=conf).getOrCreate()

print("Spark session initialized in cluster mode.")

# Step 2: Load and Clean Data
data_path = "s3://winepredictionabdeali/TrainingDataset.csv"  # Updated S3 path for training data
data = spark.read.csv(data_path, header=True, inferSchema=True, sep=";")
data = data.toDF(*[col.strip().replace('"', '') for col in data.columns])  # Clean column names
print(f"Data loaded from {data_path} with {data.count()} rows and {len(data.columns)} columns.")

# Step 3: Train-Test Split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Step 4: Feature Engineering
feature_cols = [col for col in train_data.columns if col != "quality"]

# Assemble the features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)

# Step 5: Model and Hyperparameter Tuning (Optimized)
# RandomForest Classifier
rf = RandomForestClassifier(labelCol="quality", featuresCol="scaled_features", seed=42)

# DecisionTree Classifier
dt = DecisionTreeClassifier(labelCol="quality", featuresCol="scaled_features", seed=42)

# Logistic Regression
lr = LogisticRegression(labelCol="quality", featuresCol="scaled_features", family="multinomial", maxIter=10, regParam=0.01, elasticNetParam=0.8)

# Parameter Grid for RandomForest (multi-class)
paramGrid_rf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20, 50]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .addGrid(rf.impurity, ['gini', 'entropy']) \
    .build()

# Parameter Grid for DecisionTree (multi-class)
paramGrid_dt = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [5, 10, 15]) \
    .addGrid(dt.maxBins, [16, 32]) \
    .build()

# Parameter Grid for Logistic Regression
paramGrid_lr = ParamGridBuilder() \
    .addGrid(lr.maxIter, [10, 20]) \
    .addGrid(lr.regParam, [0.01, 0.1]) \
    .addGrid(lr.elasticNetParam, [0.8, 0.9]) \
    .build()

# Step 6: Cross-validation with more folds (5-fold cross-validation)
crossval_rf = CrossValidator(estimator=rf,
                              estimatorParamMaps=paramGrid_rf,
                              evaluator=MulticlassClassificationEvaluator(labelCol="quality", metricName="accuracy"),
                              numFolds=5)  # 5-fold cross-validation

crossval_dt = CrossValidator(estimator=dt,
                              estimatorParamMaps=paramGrid_dt,
                              evaluator=MulticlassClassificationEvaluator(labelCol="quality", metricName="accuracy"),
                              numFolds=5)  # 5-fold cross-validation

crossval_lr = CrossValidator(estimator=lr,
                              estimatorParamMaps=paramGrid_lr,
                              evaluator=MulticlassClassificationEvaluator(labelCol="quality", metricName="accuracy"),
                              numFolds=5)  # 5-fold cross-validation

# Step 7: Create Pipelines
pipeline_rf = Pipeline(stages=[assembler, scaler, crossval_rf])
pipeline_dt = Pipeline(stages=[assembler, scaler, crossval_dt])
pipeline_lr = Pipeline(stages=[assembler, scaler, crossval_lr])

# Step 8: Train Models with Cross-validation
print("Training RandomForest model with hyperparameter tuning...")
pipeline_model_rf = pipeline_rf.fit(train_data)
print("RandomForest model training completed.")

print("Training DecisionTree model with hyperparameter tuning...")
pipeline_model_dt = pipeline_dt.fit(train_data)
print("DecisionTree model training completed.")

print("Training Logistic Regression model with hyperparameter tuning...")
pipeline_model_lr = pipeline_lr.fit(train_data)
print("Logistic Regression model training completed.")

# Step 9: Evaluate Models on Test Data
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
accuracy_rf = evaluator.evaluate(pipeline_model_rf.transform(test_data))
accuracy_dt = evaluator.evaluate(pipeline_model_dt.transform(test_data))
accuracy_lr = evaluator.evaluate(pipeline_model_lr.transform(test_data))

# Step 10: Model Comparison
print(f"RandomForest accuracy: {accuracy_rf:.4f}")
print(f"DecisionTree accuracy: {accuracy_dt:.4f}")
print(f"Logistic Regression accuracy: {accuracy_lr:.4f}")

# Step 11: Choose the Best Model
best_model = None
model_name = ""
if accuracy_rf > accuracy_dt and accuracy_rf > accuracy_lr:
    best_model = pipeline_model_rf
    model_name = "RandomForestModel"
elif accuracy_dt > accuracy_rf and accuracy_dt > accuracy_lr:
    best_model = pipeline_model_dt
    model_name = "DecisionTreeModel"
else:
    best_model = pipeline_model_lr
    model_name = "LogisticRegressionModel"

# Step 12: Save Best Model to S3
s3_client = boto3.client('s3')
bucket_name = "winepredictionabdeali"
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
model_dir = f"s3a://{bucket_name}/Testing_models/{model_name}_{timestamp}"

best_model.write().overwrite().save(model_dir)
print(f"Best model saved to: {model_dir}")

# Step 13: Evaluate Final Model Metrics
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

# Step 14: Stop Spark Session
spark.stop()
print("Spark session stopped.")
