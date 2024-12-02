from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import boto3
import uuid

# Step 1: Initialize Spark Session
spark = SparkSession.builder \
    .appName("Wine_Quality_single_machine_with_scaling_and_s3") \
    .master("local[*]") \
    .getOrCreate()

# Step 2: Load and Clean Data
data = spark.read.csv("/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/data/TrainingDataset.csv", 
                      header=True, inferSchema=True, sep=";")

# Clean column names to remove extra quotes
data = data.toDF(*[col.strip().replace('"', '') for col in data.columns])

# Verify 'quality' column exists
if 'quality' not in data.columns:
    raise ValueError("The dataset must contain a 'quality' column.")

# Step 3: Train-Test Split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Step 4: Assemble Features and Labels
feature_cols = [col for col in train_data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="unscaled_features")

# Add Scaling
scaler = StandardScaler(inputCol="unscaled_features", outputCol="features", withStd=True, withMean=True)

train_data = assembler.transform(train_data)
test_data = assembler.transform(test_data)

scaler_model = scaler.fit(train_data)
train_data = scaler_model.transform(train_data).select("features", "quality")
test_data = scaler_model.transform(test_data).select("features", "quality")

# Step 5: Train Models and Select Best
models = {
    "RandomForest": RandomForestClassifier(labelCol="quality", featuresCol="features", numTrees=10),
    "DecisionTree": DecisionTreeClassifier(labelCol="quality", featuresCol="features")
}

evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")

best_model_name = None
best_model = None
best_accuracy = 0.0

for model_name, model in models.items():
    print(f"Training {model_name}...")
    trained_model = model.fit(train_data)
    predictions = trained_model.transform(test_data)
    accuracy = evaluator.evaluate(predictions)
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")
    
    if accuracy > best_accuracy:
        best_model_name = model_name
        best_model = trained_model
        best_accuracy = accuracy

if best_model is None or best_accuracy < 0.8:
    print("No model met the accuracy threshold of 80%.")
else:
    print(f"Best Model: {best_model_name} with Accuracy: {best_accuracy * 100:.2f}%")
    # Save Best Model to S3
    # Save Best Model to S3
    s3 = boto3.client('s3')
    bucket_name = "winepredictionabdeali"
    key_prefix = "Wine_models/"  # Folder path in the bucket

    model_path = f"/tmp/{best_model_name}_{uuid.uuid4().hex}.model"
    best_model.write().overwrite().save(model_path)
    s3.upload_file(model_path, bucket_name, f"{key_prefix}{best_model_name}.model")
    print(f"Saved best model to S3: s3://{bucket_name}/{key_prefix}{best_model_name}.model")


# Stop Spark Session
spark.stop()
