# MLOps Production-Ready Deep Learning Project

A complete end-to-end machine learning operations (MLOps) implementation for chest scan classification using VGG16 transfer learning, DVC pipeline orchestration, Flask web interface, and automated CI/CD deployment via Jenkins and AWS.

## 🎯 Project Overview

This project demonstrates a production-grade MLOps pipeline that combines:

- **Deep Learning Model**: VGG16 pre-trained on ImageNet with custom classification head for binary chest scan classification (Healthy vs. Coccidiosis)
- **Data Pipeline**: Automated data ingestion from Google Drive with DVC versioning
- **Model Training**: Transfer learning with data augmentation and evaluation
- **Web Interface**: Flask application with REST API for inference
- **Experiment Tracking**: MLflow + DagsHub for model versioning and metrics logging
- **CI/CD Automation**: Jenkins pipeline with GitHub webhook integration
- **Containerization**: Docker with multi-stage builds
- **Cloud Deployment**: AWS EC2, ECR, and IAM for production deployment

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Model Training Pipeline](#-model-training-pipeline)
- [Deployment Guide](#-deployment-guide)
- [Contributing](#-contributing)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git
- Docker (for containerization)
- AWS Account (for cloud deployment)
- Conda or virtualenv (for environment management)

### Local Development Setup

```bash
# Clone repository
git clone https://github.com/Shubham9975/MLOPs-Production-Ready-Deep-Learning-Project.git
cd MLOPs-Production-Ready-Deep-Learning-Project

# Create conda environment
conda create -n chest_dl_env python=3.8 -y
conda activate chest_dl_env

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Run Locally

```bash
# Start Flask application
python app.py
# Access web UI at http://localhost:8080

# Or run complete training pipeline
python main.py

# Or use DVC for orchestration
dvc repro
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│           Development Environment                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐  ┌──────────┐  ┌───────────────┐    │
│  │   Git Repo   │  │ DVC Pipe │  │  ML Training  │    │
│  │   (GitHub)   │  │  (dvc.)  │  │   (Python)    │    │
│  └──────┬───────┘  └────┬─────┘  └───────┬───────┘    │
│         │                │                │             │
│         └────────┬───────┴────────┬───────┘             │
│                  │                │                     │
│                  ▼                ▼                     │
│         ┌─────────────────────────────┐                │
│         │   MLflow + DagsHub          │                │
│         │  (Experiment Tracking)      │                │
│         └─────────────────────────────┘                │
└─────────────────────────────────────────────────────────┘
                       │
                       │ git push
                       ▼
         ┌─────────────────────────────┐
         │    GitHub Repository        │
         │  (Webhook configured)       │
         └──────────┬──────────────────┘
                    │
                    │ HTTP POST
                    ▼
         ┌─────────────────────────────┐
         │   Jenkins CI/CD Server      │
         │  (Automated Pipeline)       │
         └──────────┬──────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
    ┌───────┐  ┌────────┐  ┌──────────┐
    │ Build │  │ Login  │  │   Push   │
    │ Image │  │  ECR   │  │ to ECR   │
    └───┬───┘  └────────┘  └────┬─────┘
        │                        │
        └────────────┬───────────┘
                     │
                     ▼
         ┌─────────────────────────────┐
         │  AWS ECR Registry           │
         │  (Docker Image Store)       │
         └──────────┬──────────────────┘
                    │
                    │ SSH Deploy
                    ▼
         ┌─────────────────────────────┐
         │  AWS EC2 App Server         │
         │  Docker Container Running   │
         │  Flask API :8080            │
         └─────────────────────────────┘
                    │
                    ▼
         ┌─────────────────────────────┐
         │   Production Application    │
         │  Classification API Online  │
         └─────────────────────────────┘
```

## 📁 Project Structure

```
.
├── artifacts/                          # Training outputs and models
│   ├── data_ingestion/                # Downloaded dataset
│   │   └── Chest-CT-Scan-data/
│   │       ├── adenocarcinoma/        # Disease class
│   │       └── normal/                # Healthy class
│   ├── prepare_base_model/            # VGG16 models
│   │   ├── base_model.h5
│   │   └── base_model_updated.h5
│   └── training/                       # Trained model
│       └── model.h5
│
├── config/                             # Configuration files
│   └── config.yaml                    # Paths and artifact configs
│
├── src/cnnClassifier/                 # Main package
│   ├── components/                    # ML pipeline components
│   │   ├── data_ingestion.py
│   │   ├── prepare_base_model.py      # VGG16 setup
│   │   ├── model_trainer.py           # Training logic
│   │   └── evaluation.py              # Model evaluation
│   ├── config/                        # Configuration management
│   │   └── configuration.py
│   ├── entity/                        # Data entities
│   │   └── config_entity.py
│   ├── pipeline/                      # Orchestrated stages
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_prepare_base_model.py
│   │   ├── stage_03_model_trainer.py
│   │   ├── stage_04_evaluation.py
│   │   └── predict.py                 # Inference pipeline
│   └── utils/                         # Utilities
│       └── common.py
│
├── templates/                          # Web UI
│   └── index.html
│
├── scripts/                            # Setup scripts
│   ├── ec2_setup.sh                   # EC2 server setup
│   └── jenkins.sh                     # Jenkins server setup
│
├── research/                           # Research notebooks
│   ├── 01_data_ingestion.ipynb
│   └── trials.ipynb
│
├── .jenkins/                           # Jenkins configuration
│   └── Jenkinsfile                    # CI/CD pipeline
│
├── Dockerfile                          # Container image definition
├── docker-compose.yml                 # Container orchestration
├── dvc.yaml                           # DVC pipeline stages
├── params.yaml                        # Model hyperparameters
├── config.yaml                        # Application config
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup
├── main.py                            # Manual training script
├── app.py                             # Flask web server
├── DEPLOYMENT_GUIDE.md                # 📘 Detailed deployment docs
├── README.md                          # This file
└── LICENSE
```

## 🛠️ Setup & Installation

### Environment Setup

```bash
# Using Conda
conda create -n chest_dl python=3.8 -y
conda activate chest_dl

# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .  # Install package in development mode
```

### Configuration

Edit `config/config.yaml` for your paths:
```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: "https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=drive_link"
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion

prepare_base_model:
  root_dir: artifacts/prepare_base_model
  base_model_path: artifacts/prepare_base_model/base_model.h5
  updated_base_model_path: artifacts/prepare_base_model/base_model_updated.h5

training:
  root_dir: artifacts/training
  trained_model_path: artifacts/training/model.h5
```

### Hyperparameter Tuning

Edit `params.yaml`:
```yaml
AUGMENTATION: True
IMAGE_SIZE: [224, 224, 3]
BATCH_SIZE: 16
INCLUDE_TOP: False
EPOCHS: 1
CLASSES: 2
WEIGHTS: imagenet
LEARNING_RATE: 0.01
```

## 🎮 Usage

### Web Interface

```bash
# Start Flask application
python app.py

# Access at http://localhost:8080
# - Upload chest scan image
# - Click "Predict"
# - View classification result
```

### API Endpoints

**GET** `/` - Web UI (HTML form)

**POST** `/predict` - Predict image class
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_string"}'

# Response
{
  "image": "Healthy"  # or "Coccidiosis"
}
```

**GET/POST** `/train` - Trigger training pipeline
```bash
curl -X POST http://localhost:8080/train

# Response
"Training done successfully!"
```

### Command Line

```bash
# Run complete pipeline
python main.py

# Or use DVC for smart caching
dvc repro

# View pipeline structure
dvc dag
```

## 🔄 Model Training Pipeline

The training pipeline consists of 4 stages orchestrated by DVC:

### Stage 1: Data Ingestion
- Downloads dataset from Google Drive
- Extracts and organizes data
- Output: `artifacts/data_ingestion/Chest-CT-Scan-data/`

### Stage 2: Base Model Preparation
- Loads VGG16 pre-trained on ImageNet
- Removes top classification layers (`include_top=False`)
- Adds custom classification head (2 classes)
- Freezes all pre-trained layers
- Output: `artifacts/prepare_base_model/base_model_updated.h5`

### Stage 3: Model Training
- Loads updated base model
- Applies data augmentation (rotation, flip, zoom, etc.)
- Trains custom head on augmented data
- 80/20 train-validation split
- Output: `artifacts/training/model.h5`

### Stage 4: Evaluation
- Evaluates model on validation set
- Logs metrics to MLflow (loss, accuracy)
- Registers model in MLflow registry
- Output: `scores.json`

### Execute Pipeline

```bash
# Run all stages with dependency tracking
dvc repro

# Run specific stage
dvc repro src/cnnClassifier/pipeline/stage_03_model_trainer.py

# View execution DAG
dvc dag
```

## 🚀 Deployment Guide

**For comprehensive deployment instructions, see [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)**

The guide includes:

- **Project Overview**: Architecture, tech stack, model details
- **Project Structure**: File-by-file explanation with deployment roles
- **Model Training Pipeline**: Stage-by-stage process with dependencies
- **DVC Integration**: Pipeline orchestration and artifact management
- **Dockerization**: Complete Dockerfile analysis line-by-line
- **AWS Infrastructure**: EC2, ECR, IAM, Elastic IP configuration
- **Jenkins CI/CD Pipeline**: All stages and authentication flow
- **Complete CI/CD Flow**: End-to-end from GitHub push to production
- **Credentials Management**: All Jenkins credentials explained
- **Troubleshooting**: 4 common deployment issues with solutions
- **Verification Commands**: Docker, AWS, and testing commands
- **Future Improvements**: 10 actionable recommendations for production

### Quick Deployment Summary

```bash
# 1. Prepare repository
git add -A
git commit -m "Ready for deployment"
git push origin main

# 2. Jenkins webhook triggers automatically
# - Builds Docker image
# - Pushes to ECR
# - Deploys to application server

# 3. Access deployed application
# http://<elastic_ip>:8080
```

## 📊 Tech Stack Details

| Component | Technology | Version |
|-----------|-----------|---------|
| Deep Learning | TensorFlow/Keras | 2.12.0 |
| Web Framework | Flask | Latest |
| Containerization | Docker | Latest |
| Pipeline Orchestration | DVC | Latest |
| Experiment Tracking | MLflow | 2.2.2 |
| ML Integration | DagsHub | - |
| CI/CD | Jenkins | Latest |
| Cloud Provider | AWS (EC2, ECR, IAM) | - |
| Python Version | 3.8 | - |

## 📈 Model Performance

The VGG16 transfer learning model achieves:
- **Architecture**: VGG16 base + custom classification head
- **Input**: 224x224x3 RGB images
- **Output**: Binary classification (Healthy/Coccidiosis)
- **Pre-training**: ImageNet weights
- **Strategy**: Frozen convolutional base + trained classifier head
- **Data Augmentation**: Yes (rotation, flip, zoom, shift)
- **Metrics tracked**: Loss and accuracy via MLflow

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see [LICENSE](./LICENSE) file for details.

## 👤 Author

**Shubham Rampurkar**
- GitHub: [@Shubham9975](https://github.com/Shubham9975)
- Email: rampurkarshubham91@gmail.com

## 🔗 References

- [TensorFlow VGG16 Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16)
- [DVC Documentation](https://dvc.org/)
- [MLflow Documentation](https://mlflow.org/)
- [Docker Documentation](https://docs.docker.com/)
- [Jenkins Documentation](https://www.jenkins.io/doc/)

## 📚 Additional Resources

- **Deployment Guide**: [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) - Complete production deployment documentation
- **Model Architecture**: VGG16 transfer learning with custom head
- **Dataset**: Chest scan classification (Healthy vs. Disease)
- **Pipeline**: 4-stage DVC-orchestrated ML pipeline