pipeline {
    agent any
    environment {
        STANDALONE_HOST = 'ubuntu@172.31.14.243' // Standalone server
        KUBERNETES_HOST = 'ubuntu@172.31.0.8' // Kubernetes server
        REMOTE_WORK_DIR = '/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark'
        KUBERNETES_DIR = '/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/Kubernetes_manifest'
    }
    stages {
        stage('Run Training Job') {
            steps {
                script {
                    checkoutRepo(env.STANDALONE_HOST)

                    echo "Running Spark training job..."
                    def trainingLogs = sh(
                        script: """
                        ssh ${env.STANDALONE_HOST} "
                        cd ${env.REMOTE_WORK_DIR} &&
                        /opt/spark/bin/spark-submit --master local[*] --name Wine_Quality_Prediction /home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/notebook/Wine_Quality_standalone.py
                        "
                        """,
                        returnStdout: true
                    ).trim()

                    // Extract model directory from logs
                    def modelDirMatch = trainingLogs =~ /MODEL_DIR=(\S+)/
                    if (modelDirMatch) {
                        env.MODEL_DIR = modelDirMatch[0][1]
                        echo "Extracted Model Directory: ${env.MODEL_DIR}"
                    } else {
                        error("Failed to extract the model directory from training logs.")
                    }
                }
            }
        }
        stage('Run Prediction Job on Standalone') {
            steps {
                echo "Running prediction job on standalone machine..."
                sh """
                ssh ${env.STANDALONE_HOST} "
                cd ${env.REMOTE_WORK_DIR} &&
                /opt/spark/bin/spark-submit --master local[*] --deploy-mode client /home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/notebook/Wine_Prediction_standalone.py ${params.VALIDATION_PATH} ${env.MODEL_DIR}
                "
                """
            }
        }
        stage('Dockerize Validation Job') {
            steps {
                echo "Building Docker image for validation job..."
                sh """
                ssh ${env.STANDALONE_HOST} "
                cd ${env.REMOTE_WORK_DIR} &&
                sudo docker build -t ${params.Docker_Image} .
                "
                """
            }
        }
        stage('Push Docker Image') {
            steps {
                echo "Pushing Docker image to Docker Hub..."
                withCredentials([usernamePassword(credentialsId: 'dockerhub-credentials-id', 
                                                  usernameVariable: 'DOCKER_CREDENTIALS_USR', 
                                                  passwordVariable: 'DOCKER_CREDENTIALS_PSW')]) {
                    sh """
                    ssh ${env.STANDALONE_HOST} "
                    sudo docker login -u ${DOCKER_CREDENTIALS_USR} -p ${DOCKER_CREDENTIALS_PSW} &&
                    sudo docker tag ${params.Docker_Image} ${DOCKER_CREDENTIALS_USR}/${params.Docker_Image} &&
                    sudo docker push ${DOCKER_CREDENTIALS_USR}/${params.Docker_Image}
                    "
                    """
                }
            }
        }
        stage('Run Docker Image') {
            steps {
                echo "Running Docker container for validation job..."
                sh """
                ssh ${env.STANDALONE_HOST} "
                sudo docker run --rm ${params.Docker_Image} ${params.VALIDATION_PATH} ${env.MODEL_DIR} &&
                sudo docker image prune -a -f
                "
                """
            }
        }
        stage('Kubernetes Deployment') {
            steps {
                script {
                    checkoutRepo(env.KUBERNETES_HOST)

                    echo "Preparing Kubernetes deployment using read/write YAML..."

                    // Read YAML template
                    def yamlTemplate = readFile(file: "${env.KUBERNETES_DIR}/deployment.yaml")

                    // Replace placeholders dynamically
                    def yamlContent = yamlTemplate.replace("{{IMAGE}}", "${DOCKER_CREDENTIALS_USR}/${params.Docker_Image}")
                                                  .replace("{{VALIDATION_PATH}}", "${params.VALIDATION_PATH}")
                                                  .replace("{{MODEL_FOLDER}}", "${env.MODEL_DIR}")

                    // Write updated YAML to a separate file
                    writeFile file: "${env.KUBERNETES_DIR}/deployment_updated.yaml", text: yamlContent

                    // Apply the Kubernetes job
                    sh """
                    ssh ${env.KUBERNETES_HOST} "
                    cd ${env.KUBERNETES_DIR} &&
                    kubectl apply -f deployment_updated.yaml &&
                    kubectl wait --for=condition=complete --timeout=300s job/wine-quality-prediction-${BUILD_NUMBER}
                    "
                    """
                }
            }
        }
    }
}

// Reusable function to checkout the repository
def checkoutRepo(server) {
    sh """
    ssh ${server} "
    if [ -d ${env.REMOTE_WORK_DIR} ]; then
        cd ${env.REMOTE_WORK_DIR} &&
        git pull
    else
        git clone https://github.com/mewawalaabdeali/Wine_Prediction_Distributed_System_Apache_Spark.git ${env.REMOTE_WORK_DIR}
    fi
    cd ${env.REMOTE_WORK_DIR} &&
    ls -ltra
    "
    """
}
