pipeline {
    agent any
    environment {
        STANDALONE_HOST = 'ubuntu@172.31.14.243'
        REMOTE_WORK_DIR = '/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark'
        DOCKER_CREDENTIALS = credentials('dockerhub-credentials-id') // Store Docker Hub credentials in Jenkins
    }
    stages {
        stage('Checkout Repository') {
            steps {
            echo "Checking out the repository directly on the standalone machine..."
            sh """
            ssh ${env.STANDALONE_HOST} "
            if [ -d ${env.REMOTE_WORK_DIR} ]; then
                cd ${env.REMOTE_WORK_DIR}
                git pull
            else
                git clone https://github.com/mewawalaabdeali/Wine_Prediction_Distributed_System_Apache_Spark.git ${env.REMOTE_WORK_DIR}
            fi
            cd ${env.REMOTE_WORK_DIR}
            ls -ltra
            "
            """
                }
        }
        stage('Run Training Job') {
            steps {
                script {
                    echo "Running Spark training job..."
                    def trainingLogs = sh(
                        script: """
                        ssh ${env.STANDALONE_HOST} "
                        source ~/.bashrc &&
                        cd ${env.REMOTE_WORK_DIR} && 
                        /opt/spark/bin/spark-submit --master local[*] --name Wine_Quality_Prediction /home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/notebook/Wine_Quality_standalone.py
                        "
                        """,
                        returnStdout: true
                    ).trim()

                    // Extract model directory from logs
                    def modelDirMatch = trainingLogs =~ /MODEL_DIR=(\S+)/
                    if (modelDirMatch) {
                        env.MODEL_DIR = modelDirMatch[0][1] // Extract the directory name
                        echo "Extracted Model Directory: ${env.MODEL_DIR}"
                    } else {
                        error("Failed to extract the model directory from training logs.")
                    }
                }
            }
        }
        stage('Run Prediction Job') {
            steps {
                echo "Running Spark prediction job..."
                sh """
                ssh ${env.STANDALONE_HOST} "
                cd ${env.REMOTE_WORK_DIR} &&
                /opt/spark/bin/spark-submit --master local[*] --deploy-mode client /home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/notebook/Wine_Prediction_standalone.py ${params.Validation_url} ${env.MODEL_DIR}
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
                sudo docker build -t ${Docker_Image} .
                "
                """
            }
        }
        stage('Push Docker Image') {
            steps {
                echo "Pushing Docker image to Docker Hub..."
                sh """
                ssh ${env.STANDALONE_HOST} "
                sudo docker login -u ${env.DOCKER_CREDENTIALS_USR} -p ${env.DOCKER_CREDENTIALS_PSW} &&
                sudo docker tag ${params.Docker_Image} ${env.DOCKER_CREDENTIALS_USR}/${params.Docker_Image} &&
                sudo docker push ${env.DOCKER_CREDENTIALS_USR}/${params.Docker_Image}
                "
                """
            }
        }
        stage('Run Docker Image') {
            steps {
                echo "Running Docker container for validation job..."
                sh """
                ssh ${env.STANDALONE_HOST} "
                sudo docker run -it --rm ${env.DOCKER_CREDENTIALS_USR}/${params.Docker_Image} ${params.Validation_url} ${env.MODEL_DIR}
                "
                """
            }
        }
    }
}
