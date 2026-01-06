# EduPredict – Integrated Learning Analytics Platform

## Project Overview
EduPredict is an AI-driven learning analytics platform designed to transform online education from a reactive system into a proactive, student-centered ecosystem. It continuously monitors student engagement, predicts academic performance, assesses dropout risk, and delivers personalized feedback in real time.

The system integrates behavioral analytics, machine learning, sentiment analysis, and recommender systems to support both educators and students before learning issues become critical.

---

## System Architecture
The EduPredict platform consists of four interconnected components operating in a closed-loop feedback cycle:

1. Engagement Analytics Module
2. Performance Prediction Engine
3. Dropout Risk Assessment Tool
4. Personalized Feedback Generator

### Architecture Diagram
https://drive.google.com/drive/folders/12XtRLnzo82sFOA9ptgnUrbkF5TU9FjNs?usp=sharing

---

## Component Breakdown

### 1. Engagement Analytics Module
- Collects LMS interaction data (logins, views, submissions, forum activity)
- Extracts behavioral, temporal, resource-based, and social features
- Uses adaptive DBSCAN clustering to identify engagement personas
- Computes Engagement Trajectory Index (ETI)
- Sends ETI to downstream components

### 2. Performance Prediction Engine

This component implements a machine learning–based student performance prediction engine for an LMS.
-Uses academic, behavioral, and engagement data
-Trained regression models:

 -Linear Regression (selected model)
 -Random Forest Regression
 -Gradient Boosting Regression

-Model selection based on lowest prediction error
-Initial grade classification model tested but excluded due to low accuracy
-Predicts a performance score and assigns grades using score thresholds
-Implemented using Python, Scikit-learn, and FastAPI
-Uses synthetic data (real data integration planned as future work)

### 3. Dropout Risk Assessment Tool
- Combines engagement decay, sentiment analysis, and academic risk
- BERT-based sentiment analysis on forum text
- Calculates Risk Trajectory Score (RTS)
- Produces risk levels and personas in near real time

### 4. Personalized Feedback Generator
- Hybrid recommender system:
  - Content-based filtering
  - Collaborative filtering
  - Constraint-based filtering
- Generates empathetic, personalized feedback messages
- Delivers resources via LMS/email/notifications
- Tracks feedback interactions to update engagement data

---

## Technologies & Dependencies
### Programming Languages
- Python 3.10+

### Machine Learning & AI
- Scikit-learn
- XGBoost
- Leanir Regrition 
- TensorFlow / PyTorch
- HuggingFace Transformers (BERT)

### Data Processing
- Pandas
- NumPy

### Backend / APIs
- Flask / FastAPI
- REST APIs
- LMS Integration (Moodle / Canvas APIs)

### Visualization
- Matplotlib
- Seaborn
- Plotly

### Tools & Platforms
- Git & GitHub (Version Control)
- MS Planner (Project Management)
- Jupyter Notebooks

---

## Version Control & Collaboration Strategy

- Feature-based branching strategy
- Regular commits with descriptive messages
- Pull requests for merging features
- Clear contribution history for all team members
- Full commit, branch, and merge visibility for evaluation

---

## Project Management
Task planning and contribution tracking were managed using Microsoft Planner.  
The exported Planner report is submitted separately under Checklist 2.

---

## Submission Notes
This repository is shared with evaluators for PP1 evaluation.  
Please ensure full access is granted to view commits, branches, and history.

