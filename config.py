# =======================
# Base Directory Configuration
# =======================
BASE_DIR = '/home/ubuntu'  # Change based on environment (e.g., /Wine for Docker)
TRAINING_DATA_PATH = 's3://winepredictionabdeali/TrainingDataset.csv'
VALIDATION_DATA_PATH = 's3://winepredictionabdeali/ValidationDataset.csv'
MODEL_SAVE_PATH = 's3://winepredictionabdeali/Testing_models/'

# ==========================
# Cluster and Spark Configuration
# ==========================
CLUSTER_TYPE = 'EMR'  # Options: 'local', 'docker', 'emr'
SPARK_MASTER_URL = 'spark://<spark-master-url>'  # 'local' or 'emr' URL

# ========================
# Model Training Config
# ========================
MODEL_TYPE = 'RandomForest'  # Options: 'RandomForest', 'DecisionTree'
HYPERPARAMETERS = {
    'num_trees': [10, 50],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]  # Example of extra parameter for RandomForest
}

# Cross-validation parameters
NUM_FOLDS = 3

# ========================
# Docker/EMR/Kubernetes Configuration
# ========================
DOCKER_IMAGE_PATH = '/Wine/PredictionDockerImage'  # If using Docker
EMR_INSTANCE_TYPE = 'm5.xlarge'  # Example EMR instance type
K8S_NAMESPACE = 'wine-prediction'  # Kubernetes namespace

# ========================
# Logging and Monitoring
# ========================
LOG_LEVEL = 'INFO'  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_FILE_PATH = '/home/ubuntu/logs/project_log.log'
CLOUDWATCH_LOG_GROUP = '/aws/emr/wine_prediction'

# ========================
# Miscellaneous
# ========================
RANDOM_SEED = 42
BATCH_SIZE = 64  # Batch size for data loading
PREDICTION_THRESHOLD = 0.8  # Prediction threshold for classification
