# Smart Classroom Optimization using Reinforcement Learning and MLOps

## Overview

This project implements a Smart Classroom Optimization System using Reinforcement Learning (RL) and MLOps practices.

The system automatically manages:

* Lights
* Air Conditioner (AC)
* Fan Speed

based on classroom conditions such as:

* Temperature
* Student occupancy
* Energy consumption

The goal is to improve classroom comfort while reducing unnecessary energy usage.

---

# SDG Mapping

This project supports the following United Nations Sustainable Development Goals (SDGs):

## SDG 7 – Affordable and Clean Energy

The RL agent reduces unnecessary energy consumption by intelligently controlling classroom appliances.

## SDG 11 – Sustainable Cities and Communities

The system improves smart infrastructure efficiency and supports sustainable classroom management.

---

# Problem Statement

Traditional classrooms often waste electricity because lights, fans, and AC systems remain ON even when not required.

This project uses Reinforcement Learning to automatically optimize classroom energy usage while maintaining student comfort.

The RL agent learns:

* when to turn ON/OFF lights,
* when to use AC,
* how to adjust fan speed,

based on environmental conditions.

---

# Project Architecture

```text
Smart-Classroom/
│
├── .github/
│   └── workflows/
│       └── ml_pipeline.yml
│
├── client/
│   ├── src/
│   ├── package.json
│   ├── vite.config.js
│   └── index.html
│
├── server/
│   │
│   ├── ai/
│   │   ├── agent/
│   │   ├── configs/
│   │   ├── policies/
│   │   ├── plots/
│   │   ├── results/
│   │   ├── sim/
│   │   │
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── plot_results.py
│   │   ├── test.py
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── mlflow.db
│   │
│   ├── app/
│   │   ├── main.py
│   │   └── simulation_service.py
│   │
│   └── requirements.txt
│
├── README.md
└── .gitignore
```

---

# Reinforcement Learning Methodology

## Algorithm Used

Q-Learning was used because:

* the environment is discrete,
* the state space is manageable,
* Q-learning is simple and effective for small RL environments.

---

# State Space

The state contains:

* Classroom temperature
* Number of students
* Light status
* AC status
* Fan speed

---

# Action Space

| Action ID | Action             |
| --------- | ------------------ |
| 0         | Do Nothing         |
| 1         | Toggle Lights      |
| 2         | Toggle AC          |
| 3         | Increase Fan Speed |
| 4         | Decrease Fan Speed |

---

# Reward Function

The reward function is designed to:

* maximize student comfort,
* minimize energy consumption,
* avoid unnecessary appliance usage.

Higher rewards are given when:

* classroom temperature is comfortable,
* energy consumption is low.

Negative rewards are given when:

* classroom becomes too hot/cold,
* unnecessary devices remain ON.

---

# Exploration Strategy

The project uses an epsilon-greedy exploration strategy.

During training:

* the agent initially explores random actions,
* epsilon gradually decays,
* the agent slowly shifts toward exploitation.

---

# Model Versions

Two RL models were trained using different hyperparameters.

## Model V1

Basic hyperparameters:

* Lower learning rate
* Faster epsilon decay
* Fewer episodes

Files:

* `configs/qlearning_v1.yaml`
* `policies/policy_v1.pkl`
* `results/results_v1.csv`

---

## Model V2

Improved hyperparameters:

* Higher learning rate
* Better discount factor
* Slower epsilon decay
* More training episodes

Files:

* `configs/qlearning_v2.yaml`
* `policies/policy_v2.pkl`
* `results/results_v2.csv`

The final simulation uses `policy_v2.pkl` because it achieved better rewards.

---

# Baseline vs RL Comparison

The RL agent was compared against a rule-based baseline controller.

## Baseline Controller

The baseline controller uses manually defined rules:

* turn lights ON if students are present,
* turn AC ON at high temperatures,
* basic fan control.

---

# Evaluation Metrics

The following metrics were evaluated:

* Average reward
* Energy efficiency
* Classroom comfort

---

# Results

## Comparison Table

| Method   | Performance          |
| -------- | -------------------- |
| Baseline | Lower reward         |
| RL V1    | Improved performance |
| RL V2    | Best performance     |

RL V2 achieved the best overall balance between:

* comfort,
* energy efficiency,
* decision quality.

---

# Generated Outputs

## Policies

```text
policies/
├── policy_v1.pkl
└── policy_v2.pkl
```

---

## Training Results

```text
results/
├── results_v1.csv
├── results_v2.csv
└── comparison.csv
```

---

## Plots

```text
plots/
├── reward_plot_v1.png
├── reward_plot_v2.png
└── comparison_plot.png
```

---

# MLOps Implementation

This project integrates several MLOps practices.

---

# Experiment Tracking using MLflow

MLflow is used for:

* experiment tracking,
* metric logging,
* parameter logging,
* artifact storage.

Tracked information:

* reward per episode,
* epsilon decay,
* hyperparameters,
* policies,
* comparison results.

## Start MLflow UI

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

---

# Reproducibility

The project uses:

* configuration files,
* Docker,
* Git versioning,
* MLflow tracking,
* GitHub Actions.

Anyone can reproduce the experiments using the same config files.

---

# Configuration Files

```text
configs/
├── qlearning_v1.yaml
└── qlearning_v2.yaml
```

---

# Docker Support

A Dockerfile is included to ensure consistent execution across environments.

## Build Docker Image

```bash
docker build -t smart-classroom-ai .
```

## Run Docker Container

```bash
docker run smart-classroom-ai
```

---

# CI/CD using GitHub Actions

GitHub Actions is used to automate:

* evaluation,
* plot generation,
* pipeline testing.

Workflow file:

```text
.github/workflows/ml_pipeline.yml
```

---

# Experiment Versioning

Git tags were used for experiment versioning.

Tags:

* `exp-qlearning-v1`
* `exp-qlearning-v2`

---

# Monitoring Plan

If deployed in real classrooms, the following monitoring strategies would be used.

## Sensor Monitoring

Monitor:

* DHT22 temperature sensor,
* occupancy readings,
* classroom appliance states.

Alerts would be triggered if:

* sensor stops sending data,
* occupancy remains zero for unusually long periods,
* abnormal temperature spikes occur.

---

# Data Drift Monitoring

Track changes in:

* occupancy patterns,
* temperature distributions,
* classroom usage behavior.

Significant deviations would trigger investigation or retraining.

---

# Model Performance Monitoring

The system would continuously monitor:

* average reward,
* energy consumption,
* comfort efficiency.

Retraining would be triggered if:

* rewards consistently decrease,
* energy savings drop below acceptable levels.

---

# How to Run the Project

## 1. Clone Repository

```bash
git clone <your-repo-url>
```

---

# Backend Setup

Open terminal inside project root.

Go to backend folder:

```bash
cd server
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Run FastAPI backend server:

```bash
uvicorn app.main:app --reload
```

Backend runs at:

```text
http://127.0.0.1:8000
```

---

# Frontend Setup

Open another terminal.

Go to frontend folder:

```bash
cd client
```

Install frontend dependencies:

```bash
npm install
```

Run Vite frontend server:

```bash
npm run dev
```

Frontend runs at:

```text
http://localhost:5173
```

---

# Train Model V1

Go to AI folder:

```bash
cd server/ai
```

Run training:

```bash
python train.py --config configs/qlearning_v1.yaml
```

---

# Train Model V2

```bash
python train.py --config configs/qlearning_v2.yaml
```

---

# Evaluate Models

```bash
python evaluate.py
```

---

# Generate Plots

```bash
python plot_results.py
```

---

# Run MLflow Dashboard

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

---

# Docker Commands

Go to AI folder:

```bash
cd server/ai
```

Build Docker image:

```bash
docker build -t smart-classroom-ai .
```

Run Docker container:

```bash
docker run smart-classroom-ai
```

---

# CI/CD Pipeline

GitHub Actions workflow file:

```text
.github/workflows/ml_pipeline.yml
```

The workflow automatically:

* installs dependencies,
* runs evaluation,
* generates plots.

---

# Future Improvements

Possible future improvements:

* Deep Q-Networks (DQN)
* Real IoT sensor integration
* Live classroom deployment
* Cloud-based monitoring
* Real-time analytics dashboard
* Multi-room optimization

---

# Conclusion

This project demonstrates how Reinforcement Learning and MLOps practices can be combined to build an intelligent smart classroom system.

The project successfully demonstrates:

* RL training and evaluation,
* experiment tracking,
* reproducibility,
* CI/CD automation,
* Docker deployment,
* monitoring design,
* baseline vs RL comparison.

The final RL model achieved better performance compared to the rule-based baseline while improving energy efficiency and maintaining classroom comfort.
