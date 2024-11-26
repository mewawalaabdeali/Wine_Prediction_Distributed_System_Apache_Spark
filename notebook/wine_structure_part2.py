from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.evaluation import MulticlassMetrics
import boto3
import os
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Spark
conf = SparkConf().setAppName("WineQualityPrediction_MLlib")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

def read_and_clean_data(file_path, delimiter=";"):
    """Read data from CSV and clean column names."""
    logging.info(f"Reading data from {file_path}")
    df = spark.read.csv(file_path, header=True, inferSchema=True, sep=delimiter)
    for col_name in df.columns:
        df = df.withColumnRenamed(col_name, col_name.replace('"', '').strip())
    logging.info("Data schema cleaned.")
    return df

def prepare_data(df, feature_cols, label_col):
    """Prepare data for MLlib by converting to LabeledPoint."""
    def row_to_labeled_point(row):
        features = [row[col] for col in feature_cols]
        return LabeledPoint(row[label_col], features)
    
    rdd = df.rdd.map(row_to_labeled_point)
    logging.info("Data prepared for MLlib.")
    return rdd

def train_test_split(rdd, train_ratio=0.8, seed=42):
    """Split the data into train and test sets."""
    train_data, test_data = rdd.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
    logging.info(f"Train-test split: {train_data.count()} training rows, {test_data.count()} test rows.")
    return train_data, test_data

def train_and_evaluate_logistic_regression(train_data, test_data):
    """Train and evaluate a logistic regression model."""
    logging.info("Training Logistic Regression model...")
    model = LogisticRegressionWithLBFGS.train(train_data, iterations=100)

    predictions = test_data.map(lambda lp: (float(model.predict(lp.features)), lp.label))
    metrics = MulticlassMetrics(predictions)
    accuracy = metrics.accuracy
    logging.info(f"Logistic Regression Accuracy: {accuracy:.2f}")
    return model, accuracy

def train_and_evaluate_random_forest(train_data, test_data, num_trees=10, max_depth=5):
    """Train and evaluate a random forest model."""
    logging.info("Training Random Forest model...")
    model = RandomForest.trainClassifier(
        train_data, numClasses=10, categoricalFeaturesInfo={},
        numTrees=num_trees, maxDepth=max_depth, seed=42
    )
    
    predictions = test_data.map(lambda lp: (float(model.predict(lp.features)), lp.label))
    metrics = MulticlassMetrics(predictions)
    accuracy = metrics.accuracy
    logging.info(f"Random Forest Accuracy: {accuracy:.2f}")
    return model, accuracy

def save_model_to_s3(model, local_dir, s3_bucket, s3_prefix):
    """Save the model locally and upload it to S3."""
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    model_path = os.path.join(local_dir, "model")
    model.save(sc, model_path)
    
    # Upload model to S3
    s3_client = boto3.client('s3')
    for root, dirs, files in os.walk(model_path):
        for file in files:
            s3_client.upload_file(
                os.path.join(root, file), s3_bucket,
                os.path.join(s3_prefix, os.path.relpath(os.path.join(root, file), model_path))
            )
    logging.info(f"Model uploaded to S3: s3://{s3_bucket}/{s3_prefix}")

def main():
    # File locations
    train_file = "s3://winepredictionabdeali/TrainingDataset.csv"
    s3_bucket = "winepredictionabdeali"
    s3_prefix = "Wine_models"
    local_model_dir = "/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/models"
    
    # Read and preprocess data
    df = read_and_clean_data(train_file)
    feature_cols = df.columns[:-1]  # All columns except 'quality'
    label_col = "quality"
    
    # Prepare data for MLlib
    rdd = prepare_data(df, feature_cols, label_col)
    
    # Train-test split
    train_data, test_data = train_test_split(rdd)
    
    # Train and evaluate Logistic Regression
    lr_model, lr_accuracy = train_and_evaluate_logistic_regression(train_data, test_data)
    save_model_to_s3(lr_model, local_model_dir, s3_bucket, os.path.join(s3_prefix, "logistic_regression"))

    # Train and evaluate Random Forest
    rf_model, rf_accuracy = train_and_evaluate_random_forest(train_data, test_data)
    save_model_to_s3(rf_model, local_model_dir, s3_bucket, os.path.join(s3_prefix, "random_forest"))
    
    # Print metrics
    print(f"Logistic Regression Accuracy: {lr_accuracy:.2f}")
    print(f"Random Forest Accuracy: {rf_accuracy:.2f}")

if __name__ == "__main__":
    main()
