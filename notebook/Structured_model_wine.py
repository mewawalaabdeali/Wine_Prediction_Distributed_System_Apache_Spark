from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create Spark session
spark = SparkSession.builder \
    .appName("WineQualityModel") \
    .getOrCreate()

def read_and_clean_data(file_path, file_type='csv', delimiter=';'):
    """Read data from file and clean column names."""
    try:
        logging.info(f"Reading data from {file_path}")
        df = spark.read.format(file_type).option("header", "true").option("inferSchema", "true").option("sep", delimiter).load(file_path)
        for col_name in df.columns:
            df = df.withColumnRenamed(col_name, col_name.replace('"', '').strip())
        logging.info(f"Data schema after cleaning: {df.printSchema()}")
        return df
    except Exception as e:
        logging.error(f"Error reading or cleaning data: {str(e)}")
        raise

def balance_data(df, label_col="quality", majority_classes=[5, 6], minority_classes=[3, 4, 7, 8]):
    """Balance dataset through oversampling."""
    try:
        logging.info("Balancing the dataset...")
        df_majority = df.filter(col(label_col).isin(majority_classes))
        majority_count = df_majority.count()
        oversampled_dataframes = [df_majority]
        
        for minority_class in minority_classes:
            df_minority = df.filter(col(label_col) == minority_class)
            minority_count = df_minority.count()
            if minority_count > 0:
                oversample_ratio = majority_count / minority_count
                df_oversampled = df_minority.sample(withReplacement=True, fraction=oversample_ratio)
                oversampled_dataframes.append(df_oversampled)

        df_balanced = oversampled_dataframes[0]
        for oversampled_df in oversampled_dataframes[1:]:
            df_balanced = df_balanced.union(oversampled_df)
        
        logging.info("Dataset balanced successfully.")
        return df_balanced
    except Exception as e:
        logging.error(f"Error in balancing data: {str(e)}")
        raise

def preprocess_data(df, feature_cols, label_col):
    """Preprocess data by creating feature vector and scaling."""
    try:
        logging.info("Preprocessing data...")
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        df_features = assembler.transform(df)
        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
        scaler_model = scaler.fit(df_features)
        df_scaled = scaler_model.transform(df_features)
        df_final = df_scaled.select("scaledFeatures", label_col)
        logging.info("Data preprocessing complete.")
        return df_final
    except Exception as e:
        logging.error(f"Error in preprocessing data: {str(e)}")
        raise

def train_and_evaluate_model(train_data, test_data, model, label_col="quality", features_col="scaledFeatures"):
    """Train and evaluate a model."""
    try:
        logging.info(f"Training {model.__class__.__name__}...")
        trained_model = model.fit(train_data)
        predictions = trained_model.transform(test_data)

        evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        evaluator.setMetricName("f1")
        f1_score = evaluator.evaluate(predictions)

        logging.info(f"{model.__class__.__name__} Accuracy: {accuracy:.2f}")
        logging.info(f"{model.__class__.__name__} F1 Score: {f1_score:.2f}")
        return trained_model, accuracy, f1_score
    except Exception as e:
        logging.error(f"Error training or evaluating {model.__class__.__name__}: {str(e)}")
        raise

def main():
    # File locations
    train_file = "/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/data/TrainingDataset.csv"
    test_file = "/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/data/ValidationDataset.csv"

    # Read and clean data
    df_train = read_and_clean_data(train_file)
    df_test = read_and_clean_data(test_file)

    # Balance the training data
    df_train_balanced = balance_data(df_train)

    # Preprocess training and testing data
    feature_cols = df_train_balanced.columns[:-1]
    label_col = "quality"
    df_train_final = preprocess_data(df_train_balanced, feature_cols, label_col)
    df_test_final = preprocess_data(df_test, feature_cols, label_col)

    # Train and evaluate Logistic Regression model
    lr_model = LogisticRegression(featuresCol="scaledFeatures", labelCol=label_col, maxIter=200, regParam=0.1, elasticNetParam=0.5)
    _, lr_accuracy, lr_f1 = train_and_evaluate_model(df_train_final, df_test_final, lr_model)

    # Train and evaluate Random Forest model
    rf_model = RandomForestClassifier(featuresCol="scaledFeatures", labelCol=label_col, numTrees=21, maxDepth=30)
    _, rf_accuracy, rf_f1 = train_and_evaluate_model(df_train_final, df_test_final, rf_model)

    # Train and evaluate Decision Tree model
    dt_model = DecisionTreeClassifier(featuresCol="scaledFeatures", labelCol=label_col, maxDepth=5)
    _, dt_accuracy, dt_f1 = train_and_evaluate_model(df_train_final, df_test_final, dt_model)

    logging.info("All models trained and evaluated successfully.")

if __name__ == "__main__":
    main()
