# Detecting Depression and Anxiety via Social Media Text Analysis
**Capstone Project – MSc Data Science, VIT Vellore**

This project presents a large-scale, multi-class mental health text classification system designed to detect psychological conditions from social media posts. The work rigorously compares classical machine learning models with transformer-based deep learning architectures, including the domain-specialized MentalBERT, to evaluate their effectiveness in real-world mental health signal detection.

## Objective
To design and evaluate a multi-class NLP system capable of classifying social media text into seven mental health categories, and to empirically determine whether transformer-based models—particularly domain-adapted transformers—offer significant performance improvements over well-optimized classical machine learning baselines.

## Dataset
* **Source:** Publicly available Reddit mental health communities
* **Size:** 53,000+ posts
* **Language:** English
* **Task:** Multi-class classification

### Categories (7)
* Normal
* Depression
* Anxiety
* Stress
* Bipolar Disorder
* Personality Disorder
* Suicidal Ideation

To ensure fair model comparison, class imbalance was handled through controlled under-sampling during training experiments. Full dataset details are documented in the dissertation and summarized in the `data/` directory.

## Methodology Overview
The project follows a structured, research-driven workflow to ensure reproducibility, fairness, and interpretability.

### 1. Data Preprocessing & EDA
* Removal of null values and duplicate entries
* Text normalization, noise removal, and lemmatization
* Exploratory analysis revealed strong class imbalance and large variation in post length
* Stratified train–test split applied to preserve class distribution

### 2. Classical Machine Learning Baselines
* TF-IDF feature extraction (up to 30,000 n-grams)
* Eight classical models evaluated, including:
    * Logistic Regression
    * Linear SVC
    * SGD
    * Naive Bayes
* Models compared using Accuracy and Macro F1-score
* Logistic Regression (30k n-grams) emerged as the strongest classical baseline

### 3. Transformer-Based Models
Fine-tuning of pretrained transformer architectures:
* BERT
* DistilBERT
* RoBERTa
* ClinicalBERT
* MentalBERT (domain-specific)

Identical training and evaluation procedures were applied across models to ensure fair comparison.

### 4. Evaluation & Analysis
Metrics used:
* Accuracy
* Precision
* Recall
* Macro F1-score
* Confusion matrices

Additional analysis focused on:
* Minority class detection (Suicidal Ideation, Personality Disorder)
* Error patterns and misclassification behaviour
* Impact of domain-specific pre-training

## Key Results

### Best Classical Model
**Logistic Regression (30k TF-IDF)**
* Accuracy: 74.3%
* Macro F1-score: 0.681

### Best Transformer Model
**MentalBERT**
* Accuracy: 84.0%
* Macro F1-score: 0.840

**Transformer models improved Macro F1-score by ~23% (relative) over classical baselines**

Domain-specific pre-training (MentalBERT) significantly improved detection of:
* Suicidal Ideation
* Personality Disorder
* Bipolar Disorder

These gains were most pronounced in clinically complex and minority categories where keyword-based models struggled.

## Key Learnings
* Classical ML models provide interpretable and computationally efficient baselines but struggle with contextual and narrative language.
* Transformer architectures substantially improve performance by capturing:
    * contextual meaning
    * negation
    * indirect expressions
    * narrative patterns
* Domain-adapted transformers like MentalBERT offer measurable advantages over general-purpose language models in mental health text analysis.
* Automated mental health classification should be used as a supportive screening tool, not a clinical diagnostic system.

## Ethical Considerations & Limitations
* The dataset consists of anonymized, publicly available social media posts.
* Predictions are probabilistic and subject to error—especially for high-risk categories.
* The system is not intended for autonomous clinical decision-making and must be paired with human oversight.
* Results are specific to English-language Reddit data and may not generalize across platforms or cultures without further validation.

## Academic Context
This repository represents the implementation component of the MSc Data Science capstone thesis:

**“Detecting Depression and Anxiety via Social Media Text Analysis”**

VIT Vellore, School of Advanced Sciences

November 2025

The full dissertation contains detailed literature review, experimental design, statistical analysis, and discussion of results.
