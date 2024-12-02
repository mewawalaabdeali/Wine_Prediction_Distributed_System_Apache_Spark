from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import boto3
import datetime

# Step 1: Initialize Spark Session for Cluster Mode
spark = SparkSession.builder \
    .appName("Wine_Quality_Training") \
    .config("spark.master", "yarn") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .config("spark.dynamicAllocation.enabled", "true") \
    .getOrCreate()

# Step 2: Load Training Data
data = spark.read.csv("/home/hadoop/Wine_Prediction_Distributed_System_Apache_Spark/data/TrainingDataset.csv", 
                      header=True, inferSchema=True, sep=";")
data = data.repartition(16)

# Clean Column Names
data = data.toDF(*[col.strip().replace('"', '') for col in data.columns])

# Verify 'quality' Column
if 'quality' not in data.columns:
    raise ValueError("The dataset must contain a 'quality' column.")

# Step 3: Train-Test Split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Step 4: Assemble and Scale Features
feature_cols = [col for col in train_data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)

train_data = assembler.transform(train_data)
train_data = scaler.fit(train_data).transform(train_data).select("features", "quality")
test_data = assembler.transform(test_data)
test_data = scaler.fit(test_data).transform(test_data).select("features", "quality")

# Step 5: Train Multiple Models
models = {
    "RandomForest": RandomForestClassifier(labelCol="quality", featuresCol="features", numTrees=10),
    "DecisionTree": DecisionTreeClassifier(labelCol="quality", featuresCol="features")
}

best_model = None
highest_accuracy = 0
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")

for model_name, model_instance in models.items():
    model = model_instance.fit(train_data)
    accuracy = evaluator.evaluate(model.transform(test_data))
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")
    
    if accuracy >= 0.8 and accuracy > highest_accuracy:
        best_model = model
        highest_accuracy = accuracy
        best_model_name = model_name

if not best_model:
    raise ValueError("No model met the accuracy threshold of 80%.")

# Step 6: Save Best Model to S3
model_path = f"/tmp/{best_model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.model"
best_model.write().overwrite().save(model_path)

s3 = boto3.client('s3')
s3.upload_file(Filename=model_path, Bucket="your-s3-bucket", Key=f"models/{best_model_name}.model")

# Stop Spark Session
spark.stop()
