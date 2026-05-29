# MLOps Production-Ready Deep Learning Project - Deployment Guide

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Model Training Pipeline](#3-model-training-pipeline)
4. [DVC Integration](#4-dvc-integration)
5. [Dockerization](#5-dockerization)
6. [AWS Infrastructure Used](#6-aws-infrastructure-used)
7. [Jenkins Setup](#7-jenkins-setup)
8. [Complete CI/CD Flow](#8-complete-cicd-flow)
9. [Credentials Used](#9-credentials-used)
10. [Deployment Troubleshooting](#10-deployment-troubleshooting)
11. [Verification Commands](#11-verification-commands)
12. [Future Improvements](#12-future-improvements)

---

## 1. Project Overview

### Purpose of the Project
This project implements an end-to-end machine learning operations (MLOps) pipeline for chest scan classification. It serves as a production-ready example of deploying deep learning models using modern DevOps practices, including containerization, CI/CD automation, and cloud deployment.

### Chest Scan Classification Use Case
The application classifies chest CT scans or X-ray images into two categories:
- **Healthy**: Normal chest scan with no abnormalities
- **Coccidiosis**: Abnormal chest scan indicating disease

The model is trained on a labeled dataset downloaded from Google Drive and serves predictions via a Flask web interface.

### Tech Stack Used

| Component | Technology | Version |
|-----------|-----------|---------|
| **Deep Learning Framework** | TensorFlow | 2.12.0 |
| **Pre-trained Model** | VGG16 (ImageNet weights) | - |
| **Web Framework** | Flask | Latest |
| **ML Orchestration** | DVC (Data Version Control) | Latest |
| **Experiment Tracking** | MLflow | 2.2.2 |
| **Experiment Logging** | DagsHub | - |
| **Containerization** | Docker | Latest |
| **Container Orchestration** | Docker Compose | Latest |
| **CI/CD Pipeline** | Jenkins | Latest |
| **Cloud Infrastructure** | AWS (EC2, ECR, IAM) | - |
| **Programming Language** | Python | 3.8 |
| **OS** | Linux (Debian Buster via Docker) | - |

### Model Architecture Used - VGG16 Transfer Learning

The project uses **VGG16** pre-trained on ImageNet with the following architecture:

1. **Pre-trained VGG16 Base**: 
   - Input shape: (224, 224, 3) as per VGG16 specifications
   - Pre-trained weights: ImageNet weights downloaded automatically
   - Top layers: Removed (`include_top=False`)

2. **Custom Classification Head**:
   - Flatten layer applied to VGG16 output
   - Dense layer with softmax activation for 2-class classification (Healthy/Coccidiosis)
   - SGD optimizer with learning rate of 0.01
   - Categorical cross-entropy loss

3. **Transfer Learning Strategy**:
   - **Freeze all pre-trained layers**: `freeze_all=True`
   - **Training approach**: Only the custom dense layer is trained initially
   - **Rationale**: Leverages learned features from ImageNet to reduce training time and data requirements

### Flask Application Overview

The Flask application (`app.py`) serves as the production inference server and training trigger:

**Endpoints**:
1. **GET `/`**: Serves the web UI (HTML form for image upload)
2. **GET/POST `/train`**: Triggers training via `dvc repro` command
3. **POST `/predict`**: Accepts base64-encoded image and returns classification result

**Features**:
- CORS-enabled for cross-origin requests
- Image decoding from base64 format
- Model inference with image preprocessing (224x224 resize)
- Runs on `0.0.0.0:8080` (accessible from all network interfaces)

### DVC Usage

**Data Version Control** is used to:
- Define and orchestrate multi-stage training pipeline
- Track data and model artifacts
- Create reproducible ML workflows
- Enable `dvc repro` for automatic pipeline execution

### MLflow and DagsHub Integration

- **MLflow**: Logs hyperparameters, metrics, and trained models
- **DagsHub**: Provides remote MLflow tracking server
- **Purpose**: Experiment tracking, model versioning, and performance monitoring
- **Model Registration**: VGG16 models registered as "VGG16Model" in MLflow registry

---

## 2. Project Structure

```
project_root/
├── artifacts/                 # Training outputs and model artifacts
│   ├── data_ingestion/       # Downloaded and extracted training data
│   │   └── Chest-CT-Scan-data/
│   │       ├── adenocarcinoma/    # Abnormal chest scans
│   │       └── normal/            # Healthy chest scans
│   ├── prepare_base_model/   # VGG16 base models
│   │   ├── base_model.h5           # Original VGG16 model
│   │   └── base_model_updated.h5   # VGG16 with custom head
│   └── training/             # Trained model
│       └── model.h5               # Final trained model for inference
│
├── config/                    # Configuration files
│   └── config.yaml           # Paths and artifact locations
│
├── logs/                      # Application and training logs
│   └── running_logs.log      # Execution logs from pipeline
│
├── MLOps_Tool_Demo/          # Example pipeline implementation
│   ├── demo.py               # Demonstration script
│   ├── dvc.yaml              # Example DVC stages
│   ├── pipeline/
│   │   ├── stage_01.py       # Data processing example
│   │   ├── stage_02.py       # Feature engineering example
│   │   └── stage_03.py       # Model training example
│   ├── artifacts/            # Demo outputs
│   └── requirements.txt       # Demo dependencies
│
├── research/                  # Research and experimentation
│   ├── 01_data_ingestion.ipynb    # Data exploration notebook
│   ├── trials.ipynb               # Model experimentation
│   └── test.yaml                  # Test configurations
│
├── scripts/                   # Utility scripts
│   ├── ec2_setup.sh          # AWS EC2 instance setup script
│   └── jenkins.sh            # Jenkins server setup script
│
├── src/                       # Main application source code
│   └── cnnClassifier/         # Python package for CNN classifier
│       ├── __init__.py        # Package initialization with logger setup
│       ├── components/        # ML pipeline components
│       │   ├── __init__.py
│       │   ├── data_ingestion.py       # Downloads and extracts data from Google Drive
│       │   ├── prepare_base_model.py   # Loads VGG16, adds custom head
│       │   ├── model_trainer.py        # Trains the model with data augmentation
│       │   └── evaluation.py           # Evaluates model, logs to MLflow
│       ├── config/            # Configuration management
│       │   ├── __init__.py
│       │   └── configuration.py         # Loads config.yaml and creates entity objects
│       ├── constants/         # Constants (empty in this project)
│       ├── entity/            # Data entities/dataclasses
│       │   ├── __init__.py
│       │   └── config_entity.py        # Config dataclasses (paths, params)
│       ├── pipeline/          # ML pipeline stages (orchestrated by DVC)
│       │   ├── __init__.py
│       │   ├── predict.py              # Loads model and performs inference
│       │   ├── stage_01_data_ingestion.py       # Calls DataIngestion component
│       │   ├── stage_02_prepare_base_model.py   # Calls PrepareBaseModel component
│       │   ├── stage_03_model_trainer.py        # Calls Training component
│       │   └── stage_04_evaluation.py           # Calls Evaluation component
│       └── utils/             # Utility functions
│           ├── __init__.py
│           └── common.py               # Image decoding, JSON utilities
│
├── templates/                 # Web UI templates
│   └── index.html            # Flask frontend for image upload and prediction
│
├── cnnClassifier.egg-info/   # Package metadata
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   └── top_level.txt
│
├── app.py                    # Flask application (production server)
├── main.py                   # Manual training script (calls each pipeline stage)
├── demo.py                   # Demo script for quick testing
├── Dockerfile                # Docker container definition
├── docker-compose.yml        # Docker Compose service definition
├── dvc.yaml                  # DVC pipeline stages configuration
├── params.yaml               # Model hyperparameters
├── config.yaml               # Artifact paths and configurations
├── requirements.txt          # Python package dependencies
├── setup.py                  # Package setup for installation (-e .)
├── scores.json               # Model evaluation metrics (loss, accuracy)
├── README.md                 # Project documentation
├── LICENSE                   # Project license
└── .jenkins/
    └── Jenkinsfile          # Jenkins CI/CD pipeline definition
```

### File Purposes and Deployment Participation

| File/Folder | Purpose | Deployment Role |
|------------|---------|-----------------|
| **artifacts/** | Stores trained models and data | Copied into Docker image; model.h5 needed for inference |
| **config/config.yaml** | Artifact paths, data source URLs | Configures where pipeline writes outputs |
| **src/cnnClassifier/** | Core ML package | Installed in container via `pip install -e .` |
| **templates/index.html** | Web UI | Served by Flask in production |
| **Dockerfile** | Container specification | Defines production image |
| **docker-compose.yml** | Service orchestration | Deploys container on application server |
| **app.py** | Flask web server | Main entrypoint (CMD in Dockerfile) |
| **requirements.txt** | Python dependencies | Installed in container via `pip install` |
| **dvc.yaml** | ML pipeline definition | Triggered during training |
| **params.yaml** | Model hyperparameters | Read during training and evaluation |
| **setup.py** | Package metadata | Installs cnnClassifier package |
| **.jenkins/Jenkinsfile** | CI/CD pipeline | Orchestrates build and deployment |

---

## 3. Model Training Pipeline

### Training Pipeline Overview

The training pipeline consists of 4 stages orchestrated by DVC, each represented by a Python script in `src/cnnClassifier/pipeline/`:

```
Stage 1: Data Ingestion
    ↓
Stage 2: Base Model Preparation
    ↓
Stage 3: Model Training
    ↓
Stage 4: Evaluation
```

### Stage 1: Data Ingestion

**File**: `src/cnnClassifier/components/data_ingestion.py`  
**Orchestrator**: `src/cnnClassifier/pipeline/stage_01_data_ingestion.py`

**Process**:
1. **Download Dataset**:
   - Source: Google Drive URL specified in `config/config.yaml`
   - File ID extracted from URL
   - Uses `gdown` library for secure download
   - Destination: `artifacts/data_ingestion/data.zip`

2. **Extract Data**:
   - Unzips downloaded file to `artifacts/data_ingestion/`
   - Creates directory structure:
     ```
     Chest-CT-Scan-data/
     ├── adenocarcinoma/   (abnormal/disease class - Label: 0)
     └── normal/           (healthy class - Label: 1)
     ```

3. **DVC Dependencies**:
   - Input: `config/config.yaml` (source URL)
   - Output: `artifacts/data_ingestion/Chest-CT-Scan-data/` (versioned by DVC)

### Stage 2: Base Model Preparation

**File**: `src/cnnClassifier/components/prepare_base_model.py`  
**Orchestrator**: `src/cnnClassifier/pipeline/stage_02_prepare_base_model.py`

**VGG16 Architecture Details**:

1. **Load Pre-trained VGG16**:
   ```python
   tf.keras.applications.vgg16.VGG16(
       input_shape=(224, 224, 3),
       weights='imagenet',           # Pre-trained on ImageNet
       include_top=False             # Remove top classification layer
   )
   ```
   - Automatically downloads ImageNet weights (528 MB)
   - Removes final Dense layers, keeps convolutional base

2. **Freeze Pre-trained Layers**:
   ```python
   for layer in model.layers:
       layer.trainable = False       # Freeze all VGG16 layers
   ```
   - Only custom head layers are trainable
   - Prevents overfitting on small dataset
   - Leverages ImageNet feature representations

3. **Add Custom Classification Head**:
   ```python
   flatten_in = tf.keras.layers.Flatten()(model.output)
   prediction = tf.keras.layers.Dense(
       units=2,                      # 2 classes: Healthy, Coccidiosis
       activation="softmax"
   )(flatten_in)
   ```

4. **Compile Model**:
   - Optimizer: SGD with learning rate 0.01
   - Loss: Categorical cross-entropy
   - Metrics: Accuracy

5. **Outputs**:
   - `artifacts/prepare_base_model/base_model.h5` (original VGG16)
   - `artifacts/prepare_base_model/base_model_updated.h5` (with custom head)

**DVC Parameters**:
- `IMAGE_SIZE`: [224, 224, 3]
- `WEIGHTS`: "imagenet"
- `INCLUDE_TOP`: False
- `CLASSES`: 2
- `LEARNING_RATE`: 0.01

### Stage 3: Model Training

**File**: `src/cnnClassifier/components/model_trainer.py`  
**Orchestrator**: `src/cnnClassifier/pipeline/stage_03_model_trainer.py`

**Data Preparation**:

1. **Image Data Generators**:
   - Rescale: Normalize pixel values to [0, 1] via `1/255`
   - Validation split: 20% of data for validation during training
   - Image size: 224x224 (VGG16 requirement)
   - Interpolation: Bilinear

2. **Data Augmentation** (if `AUGMENTATION: True` in params.yaml):
   - Rotation range: 40 degrees
   - Horizontal flip: Enabled
   - Width shift: 20%
   - Height shift: 20%
   - Shear range: 20%
   - Zoom range: 20%
   
   **Purpose**: Increase training data diversity without collecting more images

3. **Data Generators**:
   - Validation generator: No augmentation, 20% of data
   - Training generator: With augmentation, 80% of data
   - Batch size: 16 samples per batch

**Training Process**:

1. **Calculate Steps**:
   - `steps_per_epoch = train_samples / batch_size`
   - `validation_steps = validation_samples / batch_size`

2. **Fit Model**:
   ```python
   model.fit(
       train_generator,
       epochs=1,                    # From params.yaml
       steps_per_epoch=...,
       validation_steps=...,
       validation_data=valid_generator
   )
   ```

3. **Save Trained Model**:
   - Output: `artifacts/training/model.h5`
   - Format: HDF5 (compatible with TensorFlow)

**DVC Dependencies & Outputs**:
- Dependencies: 
  - `artifacts/prepare_base_model/` (updated model with custom head)
  - `artifacts/data_ingestion/Chest-CT-Scan-data/` (training data)
- Outputs:
  - `artifacts/training/model.h5` (tracked by DVC)

**DVC Parameters Used**:
- `IMAGE_SIZE`: [224, 224, 3]
- `EPOCHS`: 1
- `BATCH_SIZE`: 16
- `AUGMENTATION`: True

### Stage 4: Evaluation

**File**: `src/cnnClassifier/components/evaluation.py`  
**Orchestrator**: `src/cnnClassifier/pipeline/stage_04_evaluation.py`

**Process**:

1. **Load Trained Model**:
   - From: `artifacts/training/model.h5`

2. **Create Validation Generator**:
   - Validation split: 30% of data (separate from training validation)
   - Same preprocessing: Normalization, image size 224x224

3. **Evaluate Model**:
   ```python
   score = model.evaluate(valid_generator)
   # Returns: [loss, accuracy]
   ```

4. **Save Metrics**:
   - Output: `scores.json`
   - Contains: Loss and accuracy values
   - Format: JSON (DVC metric)
   - Cache: False (always recompute)

5. **Log to MLflow**:
   - Logs parameters from `params.yaml`
   - Logs metrics (loss, accuracy) to MLflow
   - Registers model as "VGG16Model" in MLflow registry
   - Tracks experiment in DagsHub

**DVC Configuration** (in dvc.yaml):
```yaml
evaluation:
  cmd: python src/cnnClassifier/pipeline/stage_04_evaluation.py
  deps:
    - artifacts/training/model.h5
    - artifacts/data_ingestion/Chest-CT-Scan-data/
  metrics:
    - scores.json:
        cache: false
```

### Reproducing the Pipeline

**Manual Execution**:
```bash
python main.py                    # Executes all 4 stages sequentially
```

**DVC Orchestration**:
```bash
dvc repro                         # DVC automatically runs stages with dependencies
```

**Via Flask Endpoint**:
```bash
curl -X POST http://localhost:8080/train
```

This triggers: `os.system("dvc repro")` in Flask app

---

## 4. DVC Integration

### Why DVC is Used

DVC (Data Version Control) solves key MLOps challenges:

1. **Pipeline Reproducibility**:
   - Defines exact sequence of transformation steps
   - Ensures consistent results across environments
   - Tracks dependencies between stages

2. **Artifact Management**:
   - Versions data and model files
   - Enables rollback to previous models
   - Tracks outputs without storing in Git

3. **Experiment Tracking**:
   - Compares metrics across pipeline runs
   - Records parameters for each experiment
   - Enables hyperparameter tuning

4. **Automation**:
   - `dvc repro` reruns only affected stages (based on dependencies)
   - Caches results to avoid redundant computation

### dvc.yaml Stages

**File Location**: `dvc.yaml`

**Stage Configuration**:

```yaml
stages:
  data_ingestion:
    cmd: python src/cnnClassifier/pipeline/stage_01_data_ingestion.py
    deps:
      - src/cnnClassifier/pipeline/stage_01_data_ingestion.py
      - config/config.yaml
    outs:
      - artifacts/data_ingestion/Chest-CT-Scan-data

  prepare_base_model:
    cmd: python src/cnnClassifier/pipeline/stage_02_prepare_base_model.py
    deps:
      - src/cnnClassifier/pipeline/stage_02_prepare_base_model.py
      - config/config.yaml
    params:
      - IMAGE_SIZE
      - INCLUDE_TOP
      - CLASSES
      - WEIGHTS
      - LEARNING_RATE
    outs:
      - artifacts/prepare_base_model

  training:
    cmd: python src/cnnClassifier/pipeline/stage_03_model_trainer.py
    deps:
      - src/cnnClassifier/pipeline/stage_03_model_trainer.py
      - config/config.yaml
      - artifacts/data_ingestion/Chest-CT-Scan-data
      - artifacts/prepare_base_model
    params:
      - IMAGE_SIZE
      - EPOCHS
      - BATCH_SIZE
      - AUGMENTATION
    outs:
      - artifacts/training/model.h5

  evaluation:
    cmd: python src/cnnClassifier/pipeline/stage_04_evaluation.py
    deps:
      - src/cnnClassifier/pipeline/stage_04_evaluation.py
      - config/config.yaml
      - artifacts/data_ingestion/Chest-CT-Scan-data
      - artifacts/training/model.h5
    params:
      - IMAGE_SIZE
      - BATCH_SIZE
    metrics:
      - scores.json:
          cache: false
```

### dvc repro Workflow

**Execution Flow**:

```
$ dvc repro
    ↓
DVC reads dvc.yaml
    ↓
Checks dependencies for each stage
    ↓
For each stage:
  ├─ If deps/params unchanged → Skip (use cache)
  ├─ If deps/params changed → Execute stage
  ├─ Record new outputs
  └─ Update dvc.lock
    ↓
Generate dvc.lock file (lock file)
```

**Advantages**:
- Only reruns affected stages (not entire pipeline)
- Caches intermediate results
- Ensures reproducibility across runs
- Enables parallel execution of independent stages

### dvc.lock Purpose

**File**: `dvc.lock` (auto-generated, not in repository)

**Contains**:
- Checksums (SHA256) of all dependencies
- Checksums of all outputs
- Command executed for each stage
- Parameter values used

**Example Entry**:
```yaml
training:
  cmd: python src/cnnClassifier/pipeline/stage_03_model_trainer.py
  deps:
  - path: artifacts/prepare_base_model
    hash: md5
    md5: abc123...
    size: 50000000
    nfiles: 2
  outs:
  - path: artifacts/training/model.h5
    hash: md5
    md5: xyz789...
    size: 100000000
```

**Purpose**:
- Ensures reproducibility (exact dependency versions)
- Enables efficient caching
- Tracks when pipeline needs rerun
- Supports `dvc dag` visualization

### Artifact Tracking

**DVC Tracked Artifacts**:
1. `artifacts/data_ingestion/Chest-CT-Scan-data/` - Training dataset
2. `artifacts/prepare_base_model/` - Base models (base_model.h5, base_model_updated.h5)
3. `artifacts/training/model.h5` - Trained model
4. `scores.json` - Evaluation metrics

**Storage**:
- Local: Stored in `.dvc/cache/`
- Remote: Can push to S3, GCS, Azure Blob Storage
- Git: Only checksums committed (`.gitignore` excludes artifacts)

**Advantages**:
- Models and data not stored in Git (keeps repository small)
- Enables collaboration via shared artifact storage
- Versioning without massive Git history

---

## 5. Dockerization

### Dockerfile Analysis

**File**: `Dockerfile`

```dockerfile
FROM python:3.8-slim

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
```

### Instruction-by-Instruction Explanation

#### 1. Base Image Selection

```dockerfile
FROM python:3.8-slim
```

**What it does**:
- Starts from official Python 3.8 slim image
- Slim variant: ~150 MB (vs. 900 MB for standard)
- Includes minimal system packages

**Why each is used**:
- **Python 3.8**: Version specified in project (compatible with TensorFlow 2.12.0)
- **Slim variant**: Reduces image size, faster deployment, fewer vulnerabilities
- **Official image**: Maintained by Docker, security updates

**Alternative Approaches**:
- `python:3.8` - Includes build tools (larger, ~300 MB)
- `python:3.8-alpine` - Extremely small (~50 MB) but missing system dependencies
- `tensorflow:2.12.0` - Includes TensorFlow pre-installed (much larger)

#### 2. System Dependencies Installation

```dockerfile
RUN apt update -y && apt install awscli -y
```

**What it does**:
- Updates apt package manager
- Installs AWS CLI (command-line interface)

**Why each is used**:
- **apt update**: Gets latest package list
- **awscli**: Needed for ECR authentication in Jenkins deployment
  - Jenkins stage "Login to ECR" uses: `aws ecr get-login-password`
  - Container must have `aws` command available
- **-y flag**: Auto-confirm installation (for non-interactive builds)

**Dependencies explanation**:
- Lightweight alternative to full Docker image with AWS tools
- Only installs what's needed for deployment

#### 3. Working Directory

```dockerfile
WORKDIR /app
```

**What it does**:
- Sets container's working directory to `/app`
- Creates `/app` if it doesn't exist
- All subsequent commands execute in this directory

**Why it's used**:
- Organizes application files in container
- Simplifies relative paths in subsequent commands
- Standard practice (not `/` or `/root`)

#### 4. Copy Project Files

```dockerfile
COPY . /app
```

**What it does**:
- Copies entire project from build context (host machine) to `/app` in container
- Preserves directory structure
- Copies: source code, config, templates, requirements.txt, setup.py, etc.

**Why it's used**:
- Makes application code available inside container
- `.` represents current directory in build context
- Happens before `pip install` (files must be present for package installation)

**Critical Note**: 
- Model artifacts in `artifacts/` are included via this COPY
- If models not committed to Git, they won't be in image
- This is a common deployment issue (see Troubleshooting section)

#### 5. Dependency Installation

```dockerfile
RUN pip install -r requirements.txt
```

**What it does**:
- Installs Python packages from `requirements.txt`
- Runs `pip` inside container environment
- Happens after COPY to leverage Docker layer caching

**Why it's used**:
- Installs all runtime dependencies:
  - TensorFlow 2.12.0
  - Flask, Flask-Cors
  - DVC, MLflow
  - Data processing libraries (pandas, numpy, gdown, scipy, etc.)

**Why order matters**:
- COPY before RUN: If code changes, pip install layer is cached (faster rebuilds)
- If requirements.txt changes, only this layer is rebuilt

**Dependencies from requirements.txt**:
```
tensorflow==2.12.0        # Deep learning framework
Flask                     # Web server
Flask-Cors                # Cross-origin requests
dvc                       # Pipeline orchestration
mlflow==2.2.2             # Experiment tracking
numpy                     # Numerical computing
pandas                    # Data manipulation
gdown                     # Google Drive download
python-box==6.0.2         # Config handling
pyYAML                    # YAML parsing
joblib                    # Serialization
scipy                     # Scientific computing
dagshub                   # MLflow integration
-e .                      # Install cnnClassifier package (editable mode)
```

#### 6. Application Startup

```dockerfile
CMD ["python3", "app.py"]
```

**What it does**:
- Sets default command to run when container starts
- Launches Flask application at `app.py`
- Maps to `python3 app.py` shell command

**Why it's used**:
- Flask app serves HTTP requests on `0.0.0.0:8080`
- Provides `/`, `/train`, `/predict` endpoints
- Runs in foreground (doesn't daemonize)

**Flask app behavior**:
```python
app.run(host='0.0.0.0', port=8080, debug=True)
```
- `host='0.0.0.0'`: Listens on all network interfaces (container-to-host accessible)
- `port=8080`: Exposed port (mapped in docker-compose.yml)
- `debug=True`: Auto-reload on code changes (not ideal for production, but acceptable for demo)

### Docker Build Context

**Included in image**:
- All source code (`src/`, `templates/`, etc.)
- Configuration files (`config.yaml`, `params.yaml`, `dvc.yaml`)
- Model artifacts (if committed and in build context)
- Requirements and setup files

**Excluded from image** (via .gitignore if used):
- `.git/` directory
- Large build artifacts
- Temporary files

### Image Layers and Caching

Docker builds images in layers:
```
Layer 1: FROM python:3.8-slim (cached, reused)
Layer 2: apt update && apt install awscli (cached)
Layer 3: WORKDIR /app (lightweight)
Layer 4: COPY . /app (invalidated if project changes)
Layer 5: pip install -r requirements.txt (invalidated if requirements.txt changes)
Layer 6: CMD ["python3", "app.py"] (metadata layer)
```

**Optimization**: Placing COPY after RUN ensures dependencies don't rebuild if only code changes.

---

## 6. AWS Infrastructure Used

### AWS Services Architecture

```
┌─────────────────────────────────────────────────┐
│              AWS Infrastructure                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────────────────┐  ┌─────────────────┐ │
│  │  EC2 Jenkins Server  │  │  IAM User       │ │
│  │  (CI/CD Pipeline)    │──│  (Credentials)  │ │
│  │                      │  │                 │ │
│  │  - Runs Jenkins      │  └─────────────────┘ │
│  │  - Builds Docker img │          │           │
│  │  - Pushes to ECR     │          ▼           │
│  │  - SSH to app server │  ┌─────────────────┐ │
│  │                      │──│  ECR Repository │ │
│  │  Elastic IP: Static  │  │                 │ │
│  │  Security Group: SSH,│  │  - Stores       │ │
│  │  Jenkins access      │  │    Docker image │ │
│  └──────────────────────┘  │  - Latest tag   │ │
│           │                │  - Versioning   │ │
│           │                └─────────────────┘ │
│           │                       ▲            │
│           │                       │            │
│           └───────────────────────┘            │
│                    SSH                         │
│           ┌──────────────────────┐            │
│           │  EC2 App Server      │            │
│           │  (Production)        │            │
│           │                      │            │
│           │  - Runs Docker       │            │
│           │    container         │            │
│           │  - Flask app:8080    │            │
│           │  - Pulls from ECR    │            │
│           │  - docker-compose    │            │
│           │    orchestration     │            │
│           │                      │            │
│           │  Elastic IP: Static  │            │
│           │  Security Group: HTTP│            │
│           │  (8080), SSH (22)    │            │
│           └──────────────────────┘            │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Detailed Service Breakdown

#### 1. EC2 Jenkins Server

**Purpose**:
- Executes CI/CD pipeline defined in Jenkinsfile
- Builds Docker images
- Pushes images to ECR
- Triggers deployments to application server

**Capabilities**:
- Runs Jenkins service as primary workload
- Installed with Java 8, Docker, AWS CLI
- Configured with SSH access for remote deployment

**Configuration** (from `scripts/jenkins.sh`):
```bash
# Install Java 8 (Jenkins requirement)
sudo apt install openjdk-8-jdk -y

# Install Jenkins from Debian repository
sudo systemctl start jenkins
sudo systemctl enable jenkins

# Install Docker
curl -fsSL https://get.docker.com | sudo sh

# Add Jenkins user to Docker group
sudo usermod -aG docker jenkins

# Install AWS CLI
sudo apt install awscli -y

# Configure AWS credentials
aws configure
```

**Elastic IP Assignment**:
- Provides static public IP address
- Enables consistent Jenkins access
- Required for GitHub webhook configuration

**Security Group Rules**:
- Inbound SSH (22): From admin machine
- Inbound Jenkins (8080): From GitHub webhooks or admin
- Outbound: All (for Docker pulls, ECR access)

#### 2. EC2 Application Server

**Purpose**:
- Runs production Flask application in Docker container
- Serves prediction API on port 8080
- Hosts web UI for image classification

**Workload**:
- Pulls latest Docker image from ECR
- Runs `docker-compose up -d` for background execution
- Exposes port 8080 publicly via Elastic IP

**Configuration**:
- Receives SSH commands from Jenkins during deployment
- Executes: `docker login` → `docker-compose up -d`
- Auto-restarts container if it crashes (via docker-compose)

**Security Group Rules**:
- Inbound SSH (22): From Jenkins server security group
- Inbound HTTP (8080): From anywhere (public Flask API)
- Outbound: All (for ECR image pull)

**Startup on Deployment** (via Jenkins SSH):
```bash
cd /home/ubuntu/
wget https://raw.githubusercontent.com/.../docker-compose.yml
export IMAGE_NAME=${ECR_REPOSITORY}:latest
aws ecr get-login-password | docker login --username AWS --password-stdin ...
docker compose up -d
```

#### 3. ECR (Elastic Container Registry)

**Purpose**:
- Centralized Docker image repository
- Stores built Docker images tagged with Git commit SHAs or "latest"
- Provides secure authentication for container pulls

**Image Storage**:
- Repository name: From Jenkins credential `ECR_REPOSITORY`
- Image URI format: `{AWS_ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{REPO_NAME}:latest`
- Example: `123456789.dkr.ecr.us-east-1.amazonaws.com/cnn-classifier:latest`

**Authentication Flow**:
1. Jenkins gets temporary ECR credentials:
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com
   ```
2. Docker login creates temporary token valid for 12 hours
3. Jenkins uses credentials to `docker push`
4. Application server uses same process to `docker pull`

**Lifecycle**:
- New image pushed on every Jenkins build
- Old images retained for rollback capability
- "latest" tag always points to most recent image

#### 4. IAM User (AWS Credentials)

**Purpose**:
- Provides programmatic access credentials for Jenkins and EC2 servers
- Controls permissions without exposing root AWS account

**Credentials Stored in Jenkins**:
1. `AWS_ACCOUNT_ID`: AWS account number (e.g., 123456789012)
2. `AWS_ACCESS_KEY_ID`: IAM access key
3. `AWS_SECRET_ACCESS_KEY`: IAM secret key

**Permissions Required** (IAM Policy):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload"
      ],
      "Resource": "arn:aws:ecr:us-east-1:123456789:repository/cnn-classifier"
    }
  ]
}
```

**Why not EC2 IAM Roles** (future improvement):
- Current approach: Credentials stored in Jenkins
- Better approach: Attach IAM role to EC2 instances
- Avoids credential management and rotation

#### 5. Elastic IP Addresses

**For Jenkins Server**:
- Static public IP for consistent webhook delivery
- GitHub can reliably reach Jenkins when repository changes
- Example: `44.205.203.85` (from Jenkinsfile SSH)

**For Application Server**:
- Static public IP for users accessing Flask API
- DNS record can point to this IP for production domains
- Users access: `http://{elastic-ip}:8080/`

**Cost**: Both Elastic IPs incur hourly charges (~$0.005/hour)

---

## 7. Jenkins Setup

### Jenkins Installation and Configuration

**Setup Steps** (from `scripts/jenkins.sh`):

```bash
# Update system packages
sudo apt update

# Install Java 8 (required by Jenkins)
sudo apt install openjdk-8-jdk -y

# Add Jenkins repository
# https://pkg.jenkins.io/debian-stable/

# Install and start Jenkins service
sudo systemctl start jenkins
sudo systemctl enable jenkins
sudo systemctl status jenkins

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Allow Jenkins to use Docker
sudo usermod -aG docker jenkins
sudo usermod -aG docker $USER
newgrp docker

# Install AWS CLI (for ECR authentication)
sudo apt install awscli -y

# Configure AWS credentials
aws configure
# Enter: AWS Access Key ID, Secret Access Key, Region, Output format

# Get Jenkins admin password
sudo cat /var/lib/jenkins/secrets/initialAdminPassword
```

**Initial Jenkins Setup**:
1. Access Jenkins at `http://{JENKINS_IP}:8080`
2. Enter initialAdminPassword from `/var/lib/jenkins/secrets/initialAdminPassword`
3. Install suggested plugins (Pipeline, Git, AWS credentials, etc.)
4. Create admin user

### Jenkinsfile Analysis

**File**: `.jenkins/Jenkinsfile`

#### Environment Variables Setup

```groovy
environment {
    ECR_REPOSITORY = credentials('ECR_REPOSITORY')
    AWS_ACCOUNT_ID = credentials('AWS_ACCOUNT_ID')
    AWS_ACCESS_KEY_ID = credentials('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = credentials('AWS_SECRET_ACCESS_KEY')
}
```

**Credentials stored in Jenkins UI**:
- **ECR_REPOSITORY**: Full ECR repository URI (e.g., `123456789.dkr.ecr.us-east-1.amazonaws.com/cnn-classifier`)
- **AWS_ACCOUNT_ID**: AWS account ID (e.g., `123456789`)
- **AWS_ACCESS_KEY_ID**: IAM access key
- **AWS_SECRET_ACCESS_KEY**: IAM secret key

**Setup in Jenkins**:
1. Navigate to "Manage Jenkins" → "Manage Credentials"
2. Create new credentials for each value
3. Reference in Jenkinsfile via `credentials('NAME')`

### CI/CD Pipeline Stages

#### Stage 1: Continuous Integration

```groovy
stage('Continuous Integration') {
    steps {
        script {
            echo "Linting repository"
            echo "Running unit tests"
        }
    }
}
```

**Current Status**: Placeholder stage (no actual tests)

**What it represents**:
- Would run code quality checks (pylint, flake8)
- Would run unit tests (pytest)
- Would validate code structure

**Future Implementation**:
```groovy
sh 'pip install flake8'
sh 'flake8 src/'
sh 'pytest tests/'
```

#### Stage 2: Login to ECR

```groovy
stage('Login to ECR') {
    steps {
        script {
            sh 'aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com'
        }
    }
}
```

**What it does**:
1. Retrieves temporary ECR authentication token:
   ```bash
   aws ecr get-login-password --region us-east-1
   ```
   - Uses AWS credentials from environment
   - Returns temporary token valid for 12 hours
   - Region must match ECR repository region

2. Authenticates Docker with ECR:
   ```bash
   docker login --username AWS --password-stdin {ECR_ENDPOINT}
   ```
   - Username fixed as "AWS" for ECR
   - Token passed via stdin
   - Creates `~/.docker/config.json` with authentication

**Why this step**:
- ECR is private registry (not Docker Hub)
- Requires authentication before push/pull
- Token expires, re-authenticated before each build

#### Stage 3: Build Image

```groovy
stage('Build Image') {
    steps {
        script {
            sh 'docker build -t ${ECR_REPOSITORY}:latest .'
        }
    }
}
```

**What it does**:
- Builds Docker image from Dockerfile in repository root
- Tags image with ECR repository name and "latest" tag
- Executes Dockerfile instructions in order

**Build command breakdown**:
- `-t ${ECR_REPOSITORY}:latest`: Tag with repository name and "latest"
- `.`: Build context (current directory = repository root)

**Build process**:
1. Creates base layer from `python:3.8-slim`
2. Installs AWS CLI
3. Copies project files
4. Installs Python dependencies (pip install -r requirements.txt)
5. Sets entrypoint

**Example image name**: `123456789.dkr.ecr.us-east-1.amazonaws.com/cnn-classifier:latest`

**What's included in image**:
- Python 3.8 with slim base (150 MB)
- AWS CLI
- All source code
- All model artifacts (if present in repository)
- All dependencies from requirements.txt

**Issue**: If model artifacts not in Git, they won't be in Docker image (common deployment problem)

#### Stage 4: Push Image

```groovy
stage('Push Image') {
    steps {
        script {
            sh 'docker push ${ECR_REPOSITORY}:latest'
        }
    }
}
```

**What it does**:
- Uploads Docker image to ECR
- Requires successful authentication from Stage 2
- Creates new image version in ECR

**Push process**:
1. Compresses image layers
2. Uploads layers to ECR
3. Updates "latest" tag pointer
4. Makes image available for deployment

**Time**: Usually 5-10 minutes (depends on image size and network)

**Failure reasons**:
- ECR authentication failed (Stage 2)
- Insufficient EC2 storage
- Network connectivity issues
- ECR repository doesn't exist

#### Stage 5: Continuous Deployment

```groovy
stage('Continuous Deployment') {
    steps {
        sshagent(['ssh_key']) {
            sh "ssh -o StrictHostKeyChecking=no -l ubuntu 44.205.203.85 'cd /home/ubuntu/ && wget https://raw.githubusercontent.com/Shubham9975/MLOPs-Production-Ready-Deep-Learning-Project/main/docker-compose.yml && export IMAGE_NAME=${ECR_REPOSITORY}:latest && aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com && docker compose up -d '"
        }
    }
}
```

**Explanation** of SSH command:

1. **sshagent wrapper**:
   ```groovy
   sshagent(['ssh_key'])
   ```
   - Uses SSH key credentials stored in Jenkins
   - Enables SSH authentication without passwords

2. **SSH connection**:
   ```bash
   ssh -o StrictHostKeyChecking=no -l ubuntu 44.205.203.85
   ```
   - `-o StrictHostKeyChecking=no`: Skip fingerprint verification (automated environment)
   - `-l ubuntu`: SSH login as "ubuntu" user
   - `44.205.203.85`: Application server Elastic IP (hardcoded)

3. **Remote commands executed on application server**:

   a. Navigate to home directory:
   ```bash
   cd /home/ubuntu/
   ```

   b. Download latest docker-compose.yml:
   ```bash
   wget https://raw.githubusercontent.com/Shubham9975/MLOPs-Production-Ready-Deep-Learning-Project/main/docker-compose.yml
   ```
   - Uses raw GitHub URL (not blob URL which downloads HTML)
   - Latest version always downloaded (overwriting old file)

   c. Set image environment variable:
   ```bash
   export IMAGE_NAME=${ECR_REPOSITORY}:latest
   ```
   - Makes latest Docker image available to docker-compose.yml
   - docker-compose.yml references via: `${IMAGE_NAME}`

   d. Authenticate with ECR:
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com
   ```
   - Gets temporary credentials
   - Logs into ECR (required before docker pull)

   e. Deploy container:
   ```bash
   docker compose up -d
   ```
   - Reads docker-compose.yml
   - Pulls latest image from ECR
   - Starts container in background (-d flag)
   - Exposes port 8080 to host

**Deployment flow visualization**:
```
Jenkins Server                          Application Server
│                                       │
├─ SSH to app server ────────────────→ │
│                                       │ wget docker-compose.yml
│                                       │
│                                       ├─ Export IMAGE_NAME
│                                       │
│                                       ├─ aws ecr get-login-password
│                                       │
│                                       ├─ docker login to ECR
│                                       │
│                                       ├─ docker pull ${IMAGE_NAME}
│                                       │  (pulls from ECR)
│                                       │
│                                       └─ docker-compose up -d
│                                           └─ Flask running on :8080
```

### Post Build Steps

```groovy
post {
    always {
        sh 'docker system prune -f'
    }
}
```

**Purpose**: Clean up Docker resources after build

**What it does**:
- Removes unused Docker images
- Removes dangling layers
- Frees disk space on Jenkins server
- `-f` flag: Force removal without confirmation

**Why important**:
- Docker builds accumulate layers (5-10 GB per build)
- Jenkins server storage limited
- Prevents "No space left on device" errors after ~5-10 builds

---

## 8. Complete CI/CD Flow

### End-to-End Deployment Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Developer Workflow                            │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │   Developer Commits Code      │
                │   (or Model Artifacts)        │
                │                               │
                │   $ git add -A                │
                │   $ git commit -m "..."       │
                │   $ git push origin main      │
                └───────────────────────────────┘
                                │
                                ▼
              ┌─────────────────────────────────┐
              │     GitHub Repository           │
              │                                 │
              │  - Receives push notification   │
              │  - Triggers webhook             │
              └─────────────────────────────────┘
                                │
                                ▼
              ┌─────────────────────────────────┐
              │    GitHub Webhook Trigger       │
              │                                 │
              │  Sends HTTP POST to Jenkins     │
              │  Endpoint: /github-webhook/     │
              └─────────────────────────────────┘
                                │
                                ▼
              ┌─────────────────────────────────┐
              │     Jenkins Server              │
              │    (CI/CD Orchestrator)         │
              │                                 │
              │  Webhook received → Job queued  │
              └─────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
        ┌─────────────────────┐  ┌──────────────────┐
        │  Stage: CI           │  │  Stage: Build    │
        ├─────────────────────┤  ├──────────────────┤
        │ 1. Checkout code    │  │ 1. Dockerfile    │
        │                     │  │ 2. docker build  │
        │ 2. Lint (echo only) │  │ 3. Tag image     │
        │                     │  │    :latest       │
        │ 3. Run tests        │  │                  │
        │    (echo only)      │  │ Output:          │
        │                     │  │ Docker image     │
        │ Result: PASS/FAIL   │  └──────────────────┘
        └─────────────────────┘          │
                    │                    │
                    └────────┬───────────┘
                             ▼
              ┌─────────────────────────────────┐
              │  Stage: Login to ECR             │
              ├─────────────────────────────────┤
              │ 1. Get ECR credentials          │
              │    aws ecr get-login-password   │
              │                                 │
              │ 2. Authenticate Docker          │
              │    docker login ...             │
              │                                 │
              │ Result: Auth token obtained     │
              └─────────────────────────────────┘
                             │
                             ▼
              ┌─────────────────────────────────┐
              │  Stage: Push Image               │
              ├─────────────────────────────────┤
              │ 1. docker push                  │
              │    ${ECR_REPOSITORY}:latest     │
              │                                 │
              │ 2. Image uploaded to ECR        │
              │                                 │
              │ Output: Image in ECR            │
              └─────────────────────────────────┘
                             │
                             ▼
              ┌─────────────────────────────────┐
              │  Stage: CD via SSH               │
              ├─────────────────────────────────┤
              │ 1. SSH to app server            │
              │    (44.205.203.85)              │
              │                                 │
              │ 2. wget docker-compose.yml      │
              │                                 │
              │ 3. Export IMAGE_NAME=           │
              │    ${ECR_REPO}:latest           │
              │                                 │
              │ 4. Docker login to ECR          │
              │                                 │
              │ 5. docker compose up -d         │
              │                                 │
              │ Result: Container running       │
              └─────────────────────────────────┘
                             │
                             ▼
              ┌─────────────────────────────────┐
              │  Post Build: Cleanup             │
              ├─────────────────────────────────┤
              │ docker system prune -f          │
              │                                 │
              │ Frees disk space on Jenkins     │
              └─────────────────────────────────┘
                             │
                             ▼
              ┌─────────────────────────────────┐
              │     Application Live             │
              ├─────────────────────────────────┤
              │ Flask running in container      │
              │ Port: 8080                      │
              │ IP: 44.205.203.85 (Elastic IP) │
              │                                 │
              │ Available endpoints:            │
              │ GET  /                          │
              │ POST /train                     │
              │ POST /predict                   │
              └─────────────────────────────────┘
```

### Detailed Stage-by-Stage Process

**Stage: Continuous Integration**
```
Trigger: GitHub Push
├─ Checkout repository (git clone)
├─ Linting (code quality checks)
│  └─ Would run: flake8 src/
├─ Unit Tests
│  └─ Would run: pytest tests/
└─ Decision: Continue if passed
```

**Stage: Build Docker Image**
```
├─ Read Dockerfile
├─ Execute commands:
│  ├─ FROM python:3.8-slim (pull base image)
│  ├─ apt update && apt install awscli
│  ├─ COPY . /app
│  └─ pip install -r requirements.txt
├─ Tag image: ${ECR_REPOSITORY}:latest
└─ Image ready for upload
```

**Stage: Authentication**
```
├─ Get temporary credentials from AWS
│  └─ aws ecr get-login-password
├─ Create Docker auth token
├─ Store credentials in ~/.docker/config.json
└─ Docker ready for push/pull
```

**Stage: Push to ECR**
```
├─ Compress Docker image layers
├─ Upload to ECR via HTTPS
├─ Update "latest" tag
└─ Image accessible for deployment
```

**Stage: Remote Deployment**
```
SSH to 44.205.203.85 (App Server)
│
├─ Download docker-compose.yml
│  └─ wget raw.githubusercontent.com/...
│
├─ Set environment variable
│  └─ export IMAGE_NAME=...
│
├─ Authenticate with ECR
│  └─ aws ecr get-login-password | docker login
│
├─ Start services
│  ├─ docker pull ${IMAGE_NAME}
│  └─ docker-compose up -d
│     └─ Container starts
│         └─ Flask app listens on :8080
│
└─ Deployment complete
```

### Deployment Duration

**Typical Timeline**:
- GitHub push → Jenkins notification: < 10 seconds
- Build & CI: 2-3 minutes
- Push to ECR: 5-10 minutes
- Deployment via SSH: 1-2 minutes
- **Total**: ~10-15 minutes from push to live

**Performance factors**:
- Docker layer caching (faster if only code changed)
- Network bandwidth (ECR push/pull speed)
- Application server startup time
- Image size (currently ~500-800 MB)

---

## 9. Credentials Used

### Jenkins Credentials Management

All credentials stored securely in Jenkins UI (Manage Jenkins → Manage Credentials).

#### Credential 1: ECR_REPOSITORY

**Name in Jenkins**: `ECR_REPOSITORY`  
**Type**: Secret text  
**Example Value**: `123456789.dkr.ecr.us-east-1.amazonaws.com/cnn-classifier`

**Where used**:
- Docker build: `docker build -t ${ECR_REPOSITORY}:latest .`
- Docker push: `docker push ${ECR_REPOSITORY}:latest`
- Deployment: `export IMAGE_NAME=${ECR_REPOSITORY}:latest`

**Why needed**:
- Identifies which ECR registry to use
- Different AWS accounts/regions use different repositories
- Avoids hardcoding AWS-specific details

#### Credential 2: AWS_ACCOUNT_ID

**Name in Jenkins**: `AWS_ACCOUNT_ID`  
**Type**: Secret text  
**Example Value**: `123456789` (12-digit number)

**Where used**:
- ECR authentication endpoint:
  ```bash
  aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com
  ```

**Why needed**:
- ECR endpoint requires AWS account ID
- Different accounts have different registries
- Enables multi-account deployments

**Format**: AWS account IDs are 12-digit numbers without hyphens

#### Credential 3: AWS_ACCESS_KEY_ID

**Name in Jenkins**: `AWS_ACCESS_KEY_ID`  
**Type**: Secret text  
**Example Value**: `AKIAIOSFODNN7EXAMPLE`

**Where used**:
- AWS CLI authentication: `aws ecr get-login-password`
- Used with AWS_SECRET_ACCESS_KEY for API calls

**Why needed**:
- Part of AWS API authentication
- Identifies which IAM user is making the request
- Combined with secret key for signed requests

#### Credential 4: AWS_SECRET_ACCESS_KEY

**Name in Jenkins**: `AWS_SECRET_ACCESS_KEY`  
**Type**: Secret text (requires special masking)  
**Example Value**: (40-character string - never shown)

**Where used**:
- AWS CLI authentication (used with ACCESS_KEY_ID)
- Signs API requests to AWS

**Why needed**:
- Secret component of AWS API authentication
- Never should be exposed in logs
- Jenkins masks this credential in build output

#### Credential 5: ssh_key

**Name in Jenkins**: `ssh_key`  
**Type**: SSH key with username  
**Value**: Private SSH key (PEM format)

**Where used**:
```groovy
sshagent(['ssh_key']) {
    sh "ssh -o StrictHostKeyChecking=no -l ubuntu 44.205.203.85 '...'"
}
```

**Why needed**:
- Authenticates Jenkins to application server (44.205.203.85)
- Private key stored securely in Jenkins
- Enables passwordless SSH deployment

**Setup**:
1. Generate SSH key pair on Jenkins server:
   ```bash
   ssh-keygen -t rsa -b 4096 -f jenkins_deploy_key
   ```
2. Add public key to application server:
   ```bash
   cat jenkins_deploy_key.pub >> ~/.ssh/authorized_keys
   ```
3. Add private key to Jenkins credentials

### Where Credentials Are Used

```
Jenkinsfile Environment:
├─ ECR_REPOSITORY
│  └─ Used in: Build, Push, Deployment stages
├─ AWS_ACCOUNT_ID
│  └─ Used in: ECR login stage
├─ AWS_ACCESS_KEY_ID
│  └─ Used by AWS CLI (combined with SECRET)
├─ AWS_SECRET_ACCESS_KEY
│  └─ Used by AWS CLI (combined with ACCESS_KEY)
└─ ssh_key
   └─ Used in: sshagent wrapper (CD stage)
```

### Credential Security Best Practices

**Current Implementation**:
- ✅ Credentials stored in Jenkins (not in Jenkinsfile)
- ✅ Credentials masked in build logs
- ✅ SSH key-based authentication

**Improvement Suggestions**:
- ❌ AWS credentials hardcoded in environment
- ❌ Could use EC2 IAM roles instead (attached to instances)
- ❌ Could use AWS Secrets Manager for rotation

**How to Improve** (future):
```groovy
// Instead of credentials:
// Use EC2 IAM roles with assume role

stage('Login to ECR') {
    steps {
        sh '''
            # EC2 has IAM role - no explicit credentials needed
            aws ecr get-login-password --region us-east-1 | \
              docker login --username AWS --password-stdin {ENDPOINT}
        '''
    }
}
```

---

## 10. Deployment Troubleshooting

### Issue 1: Docker Build Fails with Package Not Found

**Error**:
```
E: Unable to locate package awscli
E: Couldn't find any package by glob 'awscli'
E: Couldn't find any package by regex 'awscli'
```

**Root Cause**:
- Debian Buster (base of python:3.8-slim) has outdated repositories
- Package indices stale or missing

**Original Problem**:
```dockerfile
RUN apt install awscli -y
```

**Solution**:
```dockerfile
RUN apt update -y && apt install awscli -y
```

**Why it works**:
- `apt update` fetches latest package lists
- Ensures packages are found in repositories
- Always run `apt update` before `apt install` in Dockerfile

**Alternative Solutions**:
1. Use more recent base image:
   ```dockerfile
   FROM python:3.9-slim
   ```

2. Install from pip:
   ```dockerfile
   RUN pip install awscli
   ```

3. Add repository explicitly:
   ```dockerfile
   RUN apt update -y && \
       apt install -y software-properties-common && \
       add-apt-repository -y ppa:awscli-developers/ppa && \
       apt install awscli -y
   ```

---

### Issue 2: Docker Compose Download Returns HTML Instead of YAML

**Error**:
```
docker-compose.yml is valid YAML but contains HTML tags
Error response from daemon: OCI runtime create failed...
```

**Root Cause**:
- Using GitHub blob URL: `https://github.com/user/repo/blob/main/docker-compose.yml`
- GitHub redirects blob URLs to HTML page (not raw file)
- Downloaded file is HTML, not YAML

**Original Problem**:
```bash
wget https://github.com/Shubham9975/MLOPs-Production-Ready-Deep-Learning-Project/blob/main/docker-compose.yml
```

**Solution**:
```bash
wget https://raw.githubusercontent.com/Shubham9975/MLOPs-Production-Ready-Deep-Learning-Project/main/docker-compose.yml
```

**Why it works**:
- `raw.githubusercontent.com` serves raw file content
- Returns actual YAML, not HTML
- Correct format for direct file download

**Alternative Solutions**:
1. Store docker-compose.yml in ECR:
   ```bash
   docker pull ${ECR_REPOSITORY}:docker-compose
   docker run --rm ... cat /docker-compose.yml > docker-compose.yml
   ```

2. Use AWS S3:
   ```bash
   aws s3 cp s3://my-bucket/docker-compose.yml .
   ```

3. Version docker-compose.yml in repository:
   - Commit to Git
   - Pull in deployment script

---

### Issue 3: Docker Compose YAML Parse Error

**Error**:
```
ERROR: yaml.scanner.ScannerError: while scanning for the next token
found character '\t' that cannot start any plain scalar
```

**Root Cause**:
- Old docker-compose.yml file still on EC2 from previous deployment
- Corrupted or incomplete file combined with new one
- Tab characters in YAML (must use spaces)

**Example Problem**:
```bash
# First deployment creates docker-compose.yml
cd /home/ubuntu/
wget https://raw.githubusercontent.com/.../docker-compose.yml

# Second deployment APPENDS instead of OVERWRITES
wget https://raw.githubusercontent.com/.../docker-compose.yml
# File now has content from deployment 1 + deployment 2
```

**Solution 1: Remove Old File First**:
```bash
rm -f /home/ubuntu/docker-compose.yml
wget https://raw.githubusercontent.com/.../docker-compose.yml
```

**Solution 2: Use wget -O (Overwrite)**:
```bash
wget -O /home/ubuntu/docker-compose.yml \
  https://raw.githubusercontent.com/.../docker-compose.yml
```

**Solution 3: Use curl with output redirect**:
```bash
curl -o /home/ubuntu/docker-compose.yml \
  https://raw.githubusercontent.com/.../docker-compose.yml
```

**Updated Jenkinsfile Stage**:
```groovy
stage('Continuous Deployment') {
    steps {
        sshagent(['ssh_key']) {
            sh "ssh -o StrictHostKeyChecking=no -l ubuntu 44.205.203.85 ' \
              rm -f /home/ubuntu/docker-compose.yml && \
              wget -O /home/ubuntu/docker-compose.yml \
              https://raw.githubusercontent.com/.../docker-compose.yml && \
              export IMAGE_NAME=${ECR_REPOSITORY}:latest && \
              aws ecr get-login-password ... \
              docker compose up -d \
            '"
        }
    }
}
```

---

### Issue 4: Prediction Endpoint Hangs, Returns 500 Error

**Symptoms**:
```
1. Web UI loads successfully (GET /)
2. Image upload works (file selected)
3. User clicks "Predict"
4. POST /predict request hangs indefinitely
5. Timeout after 30 seconds → 500 error
6. Container logs show no errors
```

**Root Cause**:
- Model file `artifacts/training/model.h5` missing from Docker image
- Application starts successfully (Flask doesn't validate model on startup)
- Model only loaded when POST /predict is called
- Model loading fails silently or hangs

**Why it happens**:
1. Developer trains model locally:
   ```bash
   dvc repro
   # Creates artifacts/training/model.h5
   ```

2. Developer forgets to commit model to Git:
   ```bash
   git add -A
   git commit -m "Add trained model"  # NOT DONE
   ```

3. Docker build copies repository but model isn't there:
   ```dockerfile
   COPY . /app
   # artifacts/training/model.h5 doesn't exist in repo
   ```

4. Container starts but model missing:
   ```bash
   /app/artifacts/training/model.h5  # NOT FOUND
   ```

5. User triggers prediction → Model load fails:
   ```python
   model = load_model(os.path.join("artifacts/training", "model.h5"))
   # FileNotFoundError or TensorFlow hangs
   ```

**Debugging Steps**:

1. **Check container logs**:
   ```bash
   docker logs -f {container_id}
   ```

2. **List files in container**:
   ```bash
   docker exec -it {container_id} bash
   find /app -name "*.h5"
   find /app -name "*.keras"
   ls -la /app/artifacts/training/
   ```

3. **Check Flask error response**:
   ```bash
   docker exec -it {container_id} python3 -c \
     "from cnnClassifier.pipeline.predict import PredictionPipeline; \
      p = PredictionPipeline('test.jpg'); print('OK')"
   ```

4. **Verify model file exists locally**:
   ```bash
   ls -la artifacts/training/model.h5
   ```

**Solution: Commit Model to Git**

```bash
# 1. Check .gitignore doesn't exclude .h5 files
cat .gitignore

# 2. Add model artifacts
git add artifacts/training/model.h5

# 3. Commit
git commit -m "Add trained model"

# 4. Push
git push origin main

# 5. Trigger Jenkins build
# Jenkins pulls new code with model → Docker build includes model
```

**Why DVC alternative is better**:
```bash
# Alternative: Use DVC instead of Git for models
dvc add artifacts/training/model.h5
git add artifacts/training/model.h5.dvc
git commit -m "Add model metadata"

# In Docker: Install DVC and pull models
RUN dvc pull
```

**Verification Commands**:

```bash
# On deployment server
docker exec -it {container_id} bash

# Inside container
ls -la /app/artifacts/training/model.h5

# Try loading model manually
python3 -c "from tensorflow.keras.models import load_model; \
  m = load_model('/app/artifacts/training/model.h5'); print('OK')"

# Check Flask app status
curl -X GET http://localhost:8080/
# Should return HTML form

# Test prediction with sample image
curl -X POST -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image..."}' \
  http://localhost:8080/predict
# Should return classification result
```

**Resolution**:

1. Commit model artifacts:
   ```bash
   git add artifacts/training/model.h5
   git commit -m "Fix: Add trained model for deployment"
   git push origin main
   ```

2. Trigger new Jenkins build:
   - Push to main branch → GitHub webhook → Jenkins
   - OR manually trigger Jenkins job

3. New build includes model:
   - COPY . /app → includes model.h5
   - docker build creates image with model
   - docker push pushes image to ECR
   - docker compose up pulls and runs new image

4. Verify deployment:
   ```bash
   curl http://{app_server}:8080/predict
   # Should work without 500 error
   ```

---

## 11. Verification Commands

### Docker Management Commands

#### Check Running Containers

```bash
docker ps
```

**Output shows**:
- CONTAINER ID: Unique identifier
- IMAGE: Image name and tag
- COMMAND: Entrypoint command
- STATUS: Running/Stopped/Exited
- PORTS: Port mappings

**Example**:
```
CONTAINER ID  IMAGE              COMMAND          STATUS
a1b2c3d4e5f6  repo:latest       "python3 app.py" Up 2 hours  0.0.0.0:8080->8080/tcp
```

**Use case**: Verify Flask container is running

---

#### List All Docker Images

```bash
docker images
```

**Output shows**:
- REPOSITORY: Image name
- TAG: Version tag
- IMAGE ID: Unique identifier
- SIZE: Image size on disk
- CREATED: When image was built

**Example**:
```
REPOSITORY          TAG       SIZE
123456789.dkr.ecr.us-east-1.amazonaws.com/cnn-classifier  latest  650MB
python              3.8-slim  150MB
```

**Use case**: Verify Docker image built and available

---

#### View Container Logs

```bash
# Follow logs (like tail -f)
docker logs -f {container_id}

# Show last N lines
docker logs --tail 50 {container_id}

# Show timestamps
docker logs -t {container_id}
```

**Output shows**:
- Flask startup messages
- Request logs for / /train /predict
- Error messages if any
- Python exceptions

**Example log**:
```
 * Running on http://0.0.0.0:8080
 * WARNING: This is a development server. Do not use it in production.
127.0.0.1 - - [29/May/2024 10:30:45] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [29/May/2024 10:30:50] "POST /predict HTTP/1.1" 200 -
```

**Use case**: Debug issues, verify requests being served

---

#### Execute Command in Container

```bash
# Interactive bash shell
docker exec -it {container_id} bash

# Run single command
docker exec {container_id} ls -la /app/artifacts/training/

# Run Python
docker exec {container_id} python3 -c "import tensorflow; print(tensorflow.__version__)"
```

**Use cases**:
- Check file existence
- Verify module imports
- Test model loading
- Debug application issues

**Example session**:
```bash
$ docker exec -it a1b2c3d4e5f6 bash
root@a1b2c3d4e5f6:/app#

# List models
root@a1b2c3d4e5f6:/app# find . -name "*.h5"
./artifacts/training/model.h5
./artifacts/prepare_base_model/base_model.h5
./artifacts/prepare_base_model/base_model_updated.h5

# Check TensorFlow version
root@a1b2c3d4e5f6:/app# python3 -c "import tensorflow; print(tensorflow.__version__)"
2.12.0

# Exit container
root@a1b2c3d4e5f6:/app# exit
```

---

### Model File Verification

#### Find All Model Files

```bash
# On deployment server, inside container
docker exec -it {container_id} find /app -name "*.h5"
docker exec -it {container_id} find /app -name "*.keras"
```

**Expected output**:
```
/app/artifacts/prepare_base_model/base_model.h5
/app/artifacts/prepare_base_model/base_model_updated.h5
/app/artifacts/training/model.h5
```

**If any missing**: Model artifacts not included in Docker build

---

#### Check Model File Size

```bash
docker exec -it {container_id} ls -lh /app/artifacts/training/model.h5
```

**Example**:
```
-rw-r--r-- 1 root root 58M May 29 10:00 /app/artifacts/training/model.h5
```

**Normal size**: 50-150 MB (VGG16 with weights)

---

### AWS Verification

#### List ECR Images

```bash
aws ecr describe-images \
  --repository-name cnn-classifier \
  --region us-east-1
```

**Output shows**:
- Image tags
- Image digest (SHA256)
- Image size
- Created date
- Pushed date

**Verification**: Confirms latest image in ECR

---

#### Get ECR Login Token

```bash
aws ecr get-login-password --region us-east-1 | docker login \
  --username AWS \
  --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
```

**Expected output**:
```
Login Succeeded
```

**If fails**:
- Invalid AWS credentials
- ECR repository doesn't exist
- IAM user lacks permissions

---

### Application Testing

#### Test Web UI

```bash
# Get home page
curl http://{app_server}:8080/

# Should return HTML form
```

**Expected**: HTML index.html page loads

---

#### Test Prediction Endpoint

```bash
# Create test image (base64 encoded)
base64_image=$(base64 -i test_image.jpg)

# Send prediction request
curl -X POST http://{app_server}:8080/predict \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$base64_image\"}"

# Expected response
# [{"image": "Healthy"}] or [{"image": "Coccidiosis"}]
```

---

#### Test Training Trigger

```bash
# Trigger training
curl -X POST http://{app_server}:8080/train

# Response
# "Training done successfully!"
```

**Note**: Actual training requires time (10-30 minutes depending on data)

---

### Docker Compose Verification

#### Check docker-compose.yml

```bash
# On deployment server
cat /home/ubuntu/docker-compose.yml
```

**Expected content**:
```yaml
version: '3'
services:
  application:
    image: "${IMAGE_NAME}"
    ports:
      - "8080:8080"
```

---

#### Validate YAML Syntax

```bash
docker-compose config
```

**If valid**: Shows parsed configuration  
**If invalid**: Shows YAML error

---

### Pipeline Verification

#### Check DVC Pipeline

```bash
# On development machine
dvc dag
```

**Shows**:
```
      data_ingestion
            |
  prepare_base_model
            |
        training
            |
       evaluation
```

---

#### View DVC Lock File

```bash
cat dvc.lock
```

**Contains**: Checksums of all dependencies and outputs  
**Verification**: Reproducibility is tracked

---

### Checklist for Successful Deployment

```
✓ GitHub Push
  └─ Code + model artifacts committed

✓ Jenkins Trigger
  └─ Webhook fired, job queued

✓ Continuous Integration
  └─ Linting and tests passed

✓ Docker Build
  └─ Image built successfully
  └─ ~650 MB final image size

✓ Docker Push
  └─ Image pushed to ECR
  └─ Can verify: aws ecr describe-images

✓ SSH Deployment
  └─ Connection to app server successful
  └─ docker-compose.yml downloaded
  └─ Environment variable set

✓ Docker Login
  └─ ECR authentication successful
  └─ "Login Succeeded" message

✓ Container Start
  └─ docker-compose up -d executed
  └─ Container running: docker ps

✓ Flask Running
  └─ curl http://localhost:8080/ returns HTML
  └─ Container logs show: Running on 0.0.0.0:8080

✓ Model Loaded
  └─ docker exec find /app -name "*.h5" finds model.h5
  └─ Prediction endpoint responds: POST /predict

✓ Application Live
  └─ Web UI loads: http://{elastic_ip}:8080
  └─ Predictions work
```

---

## 12. Future Improvements

### Improvement 1: Use Gunicorn Instead of Flask Development Server

**Current State**:
```python
# app.py
app.run(host='0.0.0.0', port=8080, debug=True)
```

**Issue**:
- Flask development server not production-ready
- Single-threaded, handles one request at a time
- Debug mode enabled (security risk)
- No graceful shutdown
- Poor performance

**Solution**: Use Gunicorn WSGI server

**Implementation**:

1. Add to requirements.txt:
   ```
   gunicorn==20.1.0
   ```

2. Modify app.py (remove debug):
   ```python
   if __name__ == "__main__":
       app.run(host='0.0.0.0', port=8080)  # Only for dev
   ```

3. Update Dockerfile:
   ```dockerfile
   FROM python:3.8-slim
   RUN apt update -y && apt install awscli -y
   WORKDIR /app
   COPY . /app
   RUN pip install -r requirements.txt
   CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8080", "app:app"]
   ```

**Benefits**:
- Multi-worker (4+ workers handle concurrent requests)
- Production-grade server
- Better error handling
- No debug mode vulnerabilities

---

### Improvement 2: Use EC2 IAM Roles Instead of AWS Access Keys

**Current State**:
```
Jenkins credentials:
├─ AWS_ACCESS_KEY_ID
├─ AWS_SECRET_ACCESS_KEY
└─ AWS_ACCOUNT_ID

Problem: Credentials stored in Jenkins, require rotation
```

**Solution**: Attach IAM role to EC2 instances

**Implementation**:

1. Create IAM role for Jenkins EC2:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "ecr:GetAuthorizationToken",
           "ecr:BatchGetImage",
           "ecr:PutImage",
           "ecr:InitiateLayerUpload",
           "ecr:UploadLayerPart",
           "ecr:CompleteLayerUpload"
         ],
         "Resource": "*"
       }
     ]
   }
   ```

2. Attach role to Jenkins EC2 instance (AWS Console)

3. Update Jenkinsfile (no credential references):
   ```groovy
   stage('Login to ECR') {
       steps {
           sh 'aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com'
       }
   }
   // Credentials automatically assumed from EC2 role
   ```

**Benefits**:
- No credential management in Jenkins
- Automatic credential rotation by AWS
- Credentials not exposed in logs
- Better security posture

---

### Improvement 3: Force Image Pull During Deployment

**Current State**:
```bash
docker-compose up -d
# Uses cached image if available
# May serve old image if latest tag unchanged
```

**Issue**: If ECR image updated but tag is "latest", Docker might use cached image

**Solution**: Add `pull_policy: always`

**Implementation**:

Update docker-compose.yml:
```yaml
version: '3'
services:
  application:
    image: "${IMAGE_NAME}"
    pull_policy: always          # Force pull before running
    ports:
      - "8080:8080"
```

**Benefits**:
- Always gets latest image from ECR
- Prevents stale image serving
- Ensures fresh deployment

---

### Improvement 4: Add Container Health Checks

**Current State**:
```bash
docker-compose up -d
# No verification container is healthy
# May appear running but Flask not responding
```

**Solution**: Add health check

**Implementation**:

Update docker-compose.yml:
```yaml
version: '3'
services:
  application:
    image: "${IMAGE_NAME}"
    pull_policy: always
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

**Benefits**:
- Docker knows if application is healthy
- `docker ps` shows "(healthy)" status
- Auto-restart unhealthy containers
- Better deployment monitoring

---

### Improvement 5: Add Monitoring and Logging

**Current State**:
```
Application logs only in container stdout
No centralized logging
No metrics/monitoring
```

**Solution**: Use CloudWatch or ELK stack

**Implementation Option 1: AWS CloudWatch**

1. Install CloudWatch agent in container:
   ```bash
   pip install watchtower
   ```

2. Add to app.py:
   ```python
   import logging
   import watchtower

   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   logger.addHandler(watchtower.CloudWatchLogHandler())
   ```

3. Logs automatically pushed to CloudWatch

**Implementation Option 2: ELK Stack (on-premise)**

1. Docker compose with Elasticsearch, Logstash, Kibana
2. Configure Flask to send logs to Logstash
3. Visualize in Kibana dashboard

**Benefits**:
- Centralized logging
- Historical log retention
- Search and filter capabilities
- Alerting on errors
- Performance monitoring

---

### Improvement 6: Store Model in S3/DVC Instead of Git

**Current State**:
```
Models committed to Git repository
Large file sizes (50-100 MB)
Slow Git operations
Repository bloat
```

**Solution**: Use DVC with S3 backend

**Implementation**:

1. Install DVC S3 support:
   ```bash
   pip install dvc[s3]
   ```

2. Configure DVC:
   ```bash
   dvc remote add -d myremote s3://my-bucket/dvc-store
   ```

3. Track model with DVC:
   ```bash
   dvc add artifacts/training/model.h5
   git add artifacts/training/model.h5.dvc
   git commit -m "Track model with DVC"
   ```

4. Push to S3:
   ```bash
   dvc push
   ```

5. In Docker, pull model before serving:
   ```dockerfile
   RUN dvc pull
   ```

**Benefits**:
- Models not in Git repository
- Git repository stays small (<100 MB)
- DVC handles versioning
- S3 is cheaper for large files
- Easy model rollback

---

### Improvement 7: Add Automated Tests

**Current State**:
```groovy
stage('Continuous Integration') {
    steps {
        script {
            echo "Running unit tests"  // Only echo, no actual tests
        }
    }
}
```

**Solution**: Add pytest tests

**Implementation**:

1. Create `tests/test_predict.py`:
   ```python
   import pytest
   from cnnClassifier.pipeline.predict import PredictionPipeline
   from cnnClassifier.utils.common import decodeImage
   import numpy as np
   from PIL import Image
   import io

   def test_prediction_pipeline():
       """Test prediction pipeline with dummy image"""
       # Create dummy image
       img = Image.new('RGB', (224, 224), color='red')
       img_bytes = io.BytesIO()
       img.save(img_bytes, format='JPEG')
       img_base64 = base64.b64encode(img_bytes.getvalue()).decode()
       
       # Test decode
       decodeImage(img_base64, "test.jpg")
       assert os.path.exists("test.jpg")
       
       # Test prediction
       pipeline = PredictionPipeline("test.jpg")
       result = pipeline.predict()
       assert result[0]["image"] in ["Healthy", "Coccidiosis"]
   ```

2. Add pytest to requirements.txt:
   ```
   pytest
   pytest-cov
   ```

3. Update Jenkinsfile:
   ```groovy
   stage('Continuous Integration') {
       steps {
           script {
               sh 'pip install pytest pytest-cov'
               sh 'pytest tests/ --cov=src/cnnClassifier'
           }
       }
   }
   ```

**Benefits**:
- Catches regressions early
- Documents expected behavior
- Increases confidence in deployments
- Reduces production bugs

---

### Improvement 8: Use Multi-Stage Docker Builds

**Current State**:
```dockerfile
FROM python:3.8-slim
# ... all dependencies installed
# Final image includes all build tools, cache, etc.
```

**Issue**: Final image contains unnecessary build dependencies

**Solution**: Multi-stage build

**Implementation**:
```dockerfile
# Stage 1: Build
FROM python:3.8-slim as builder
RUN apt update -y && apt install -y gcc g++ make
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Stage 2: Runtime (smaller)
FROM python:3.8-slim
RUN apt update -y && apt install awscli -y
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . /app
ENV PATH=/root/.local/bin:$PATH
CMD ["python3", "app.py"]
```

**Benefits**:
- Reduces final image size (300 MB → 150 MB)
- Faster deployment
- Smaller surface area for vulnerabilities
- Only runtime dependencies in final image

---

### Improvement 9: Add Rollback Capability

**Current State**:
```bash
# Deployment always pulls "latest" tag
# If latest is broken, no easy rollback
```

**Solution**: Tag images with version/commit SHA

**Implementation**:

1. Update Jenkinsfile:
   ```groovy
   stage('Build Image') {
       steps {
           script {
               def commit_sha = sh(script: 'git rev-parse --short HEAD', returnStdout: true).trim()
               sh "docker build -t ${ECR_REPOSITORY}:${commit_sha} ."
               sh "docker build -t ${ECR_REPOSITORY}:latest ."
               sh "docker push ${ECR_REPOSITORY}:${commit_sha}"
               sh "docker push ${ECR_REPOSITORY}:latest"
           }
       }
   }
   ```

2. Rollback to previous version:
   ```bash
   # Instead of latest
   export IMAGE_NAME=${ECR_REPOSITORY}:abc1234
   docker-compose up -d
   ```

**Benefits**:
- Easy rollback to known-good versions
- Version history in ECR
- Can compare image differences
- Safer deployments

---

### Improvement 10: Use Container Orchestration (Kubernetes)

**Current State**:
```
Single EC2 instance running single container
No auto-scaling
No high availability
Manual deployment
```

**Solution**: Use AWS ECS or Kubernetes

**Benefits**:
- Auto-scaling based on load
- Multi-region failover
- Load balancing
- Automatic updates
- Better resource utilization

**Alternative**: AWS ECS is simpler than Kubernetes for this use case

---

### Summary of Priority Improvements

| Priority | Improvement | Impact | Effort |
|----------|-------------|--------|--------|
| High | Gunicorn + EC2 IAM Roles | Security, Performance | Medium |
| High | Add health checks | Reliability | Low |
| High | S3 + DVC for models | Git size, Versioning | Medium |
| Medium | Force image pull | Deployment reliability | Low |
| Medium | Automated tests | Code quality | Medium |
| Medium | Monitoring/Logging | Debugging, Operations | Medium |
| Low | Multi-stage builds | Image size | Low |
| Low | Rollback capability | Incident response | Low |
| Low | Kubernetes | Scalability | High |

---

## Summary

This deployment guide documents the complete end-to-end MLOps pipeline for a production-ready deep learning project. The architecture combines:

- **ML Pipeline**: DVC-orchestrated 4-stage training pipeline (data ingestion → model preparation → training → evaluation)
- **Model**: VGG16 transfer learning with custom classifier head
- **Web Interface**: Flask application for training triggers and predictions
- **Containerization**: Docker with minimal python:3.8-slim base
- **CI/CD**: Jenkins-driven pipeline with GitHub webhooks
- **Cloud**: AWS infrastructure (EC2, ECR, IAM) for deployment
- **Monitoring**: MLflow + DagsHub for experiment tracking

The guide includes practical troubleshooting strategies for common deployment issues and actionable recommendations for production improvements.

