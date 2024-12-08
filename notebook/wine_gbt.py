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
from pyspark.sql.functions import col, when
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

# Step 3: Handle Imbalanced Data by Adding Weights for Each Class
# Calculate class frequencies
class_weights = data.groupBy("quality").count().withColumnRenamed("count", "class_count")

# Get the total number of samples in the dataset
total_count = data.count()

# Assign a higher weight to the minority classes
weight_df = class_weights.withColumn("weight", when(col("quality") == 5, total_count / (class_weights["class_count"] * 2))
                                                .otherwise(total_count / class_weights["class_count"]))

# Collect the weight_df into a list of rows and convert it into a dictionary
weight_dict = {row['quality']: row['weight'] for row in weight_df.collect()}

# Broadcast the weight dictionary
weight_df_broadcast = spark.sparkContext.broadcast(weight_dict)

# Add weight column to the dataset based on the class label
data = data.withColumn("weight", when(col("quality").isNotNull(), 
                                         when(col("quality") == 5, weight_df_broadcast.value[5])
                                         .otherwise(weight_df_broadcast.value[col("quality")]))
)

# Step 4: Train-Test Split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Step 5: Feature Engineering
feature_cols = [col for col in train_data.columns if col != "quality" and col != "weight"]

# Assemble the features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)

# Step 6: Model and Hyperparameter Tuning (Optimized)
# RandomForest Classifier
rf = RandomForestClassifier(labelCol="quality", featuresCol="scaled_features", weightCol="weight", seed=42)

# DecisionTree Classifier
dt = DecisionTreeClassifier(labelCol="quality", featuresCol="scaled_features", weightCol="weight", seed=42)

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

# Step 7: Cross-validation with more folds (5-fold cross-validation)
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

# Step 8: Create Pipelines
pipeline_rf = Pipeline(stages=[assembler, scaler, crossval_rf])
pipeline_dt = Pipeline(stages=[assembler, scaler, crossval_dt])
pipeline_lr = Pipeline(stages=[assembler, scaler, crossval_lr])

# Step 9: Train Models with Cross-validation
print("Training RandomForest model with hyperparameter tuning...")
pipeline_model_rf = pipeline_rf.fit(train_data)
print("RandomForest model training completed.")

print("Training DecisionTree model with hyperparameter tuning...")
pipeline_model_dt = pipeline_dt.fit(train_data)
print("DecisionTree model training completed.")

print("Training Logistic Regression model with hyperparameter tuning...")
pipeline_model_lr = pipeline_lr.fit(train_data)
print("Logistic Regression model training completed.")

# Step 10: Evaluate Models on Test Data
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
accuracy_rf = evaluator.evaluate(pipeline_model_rf.transform(test_data))
accuracy_dt = evaluator.evaluate(pipeline_model_dt.transform(test_data))
accuracy_lr = evaluator.evaluate(pipeline_model_lr.transform(test_data))

# Evaluate F1 Score, Precision, and Recall
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
f1_rf = evaluator_f1.evaluate(pipeline_model_rf.transform(test_data))
f1_dt = evaluator_f1.evaluate(pipeline_model_dt.transform(test_data))
f1_lr = evaluator_f1.evaluate(pipeline_model_lr.transform(test_data))

# Step 11: Model Comparison
print(f"RandomForest accuracy: {accuracy_rf:.4f}, F1 Score: {f1_rf:.4f}")
print(f"DecisionTree accuracy: {accuracy_dt:.4f}, F1 Score: {f1_dt:.4f}")
print(f"Logistic Regression accuracy: {accuracy_lr:.4f}, F1 Score: {f1_lr:.4f}")

# Step 12: Choose the Best Model
best_model = None
model_name = ""
if f1_rf > f1_dt and f1_rf > f1_lr:
    best_model = pipeline_model_rf
    model_name = "RandomForestModel"
elif f1_dt > f1_rf and f1_dt > f1_lr:
    best_model = pipeline_model_dt
    model_name = "DecisionTreeModel"
else:
    best_model = pipeline_model_lr
    model_name = "LogisticRegressionModel"

# Step 13: Save Best Model to S3
s3_client = boto3.client('s3')
bucket_name = "winepredictionabdeali"
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
model_dir = f"s3a://{bucket_name}/Testing_models/{model_name}_{timestamp}"

best_model.write().overwrite().save(model_dir)
print(f"Best model saved to: {model_dir}")

# Step 14: Evaluate Final Model Metrics
train_prediction = best_model.transform(train_data)
test_prediction = best_model.transform(test_data)

train_f1 = evaluator_f1.evaluate(train_prediction)
test_f1 = evaluator_f1.evaluate(test_prediction)
train_accuracy = evaluator.evaluate(train_prediction)
test_accuracy = evaluator.evaluate(test_prediction)

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
