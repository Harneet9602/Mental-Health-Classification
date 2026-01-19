# Dataset Description
This directory contains the dataset used for the capstone project
“Detecting Depression and Anxiety via Social Media Text Analysis.”

## File Overview
**Combined_Data.csv**
A consolidated dataset of social media posts collected from publicly available Reddit mental health communities, used for multi-class mental health text classification.

## Dataset Summary
* **Total samples:** 53,000+ posts
* **Language:** English
* **Platform:** Reddit
* **Data type:** User-generated text (social media posts)
* **Task:** Multi-class classification

## Mental Health Categories (7)
Each post is labeled into one of the following categories:

* **Normal** – General discussion without indicators of psychological distress
* **Depression** – Expressions of persistent sadness, hopelessness, or low mood
* **Anxiety** – Expressions of fear, worry, or tension
* **Stress** – Situational emotional overload or pressure
* **Bipolar Disorder** – Indicators of mood swings or manic–depressive patterns
* **Personality Disorder** – Patterns of emotional dysregulation or interpersonal difficulties
* **Suicidal Ideation** – Explicit or implicit expressions of self-harm or suicidal thoughts

## Data Characteristics
* Posts vary significantly in length, ranging from short statements to long narrative descriptions.
* The original dataset exhibits class imbalance, with certain categories (e.g., Normal) occurring more frequently than others.
* Class imbalance was handled during model training using controlled under-sampling to ensure fair comparison across models.

## Preprocessing Notes
Text preprocessing was performed in the modeling notebooks and includes:

* Lowercasing
* Removal of URLs, mentions, hashtags, and special characters
* Lemmatization
* Tokenization strategies adapted based on model type
* Classical ML models use TF-IDF features
* Transformer models use subword tokenization

The raw dataset is preserved in this directory to maintain reproducibility.

## Ethical Considerations
* All data originates from publicly accessible Reddit posts.
* No personal identifiers are included in the dataset.
* The dataset is used strictly for academic and research purposes.
* This dataset and associated models are not intended for clinical diagnosis or autonomous decision-making.

## Usage Disclaimer
Predictions derived from this dataset should be interpreted as supportive analytical signals, not definitive assessments of mental health conditions.

Human expertise and ethical oversight are essential for any real-world application.
