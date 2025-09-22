# Capstone Project: Multi-Class Mental Health Classification

## Project Flow

### 1. Phase 0: Setup and Environment

* All necessary libraries for data manipulation (pandas), machine learning (scikit-learn), deep learning (TensorFlow/Keras), and visualization (matplotlib/seaborn) are imported.
* A consistent random seed is set to ensure the reproducibility of all experiments.

### 2. Phase 1: Data Loading, Exploration, and Preprocessing

* The raw dataset is loaded and cleaned of any null values or duplicate entries.
* An initial exploratory data analysis (EDA) is performed to visualize the class distribution, revealing a significant imbalance.
* To address this, a balanced dataset is created by under-sampling 800 posts from each of the seven categories. This ensures the models are trained on a fair representation of the data.
* A robust text preprocessing function is defined and applied. This function handles the expansion of contractions, conversion to lowercase, removal of noise (URLs, hashtags), and lemmatization, preparing the text for vectorization.
* Finally, the preprocessed dataset is split into training (80%) and testing (20%) sets.

### 3. Phase 2: Baseline Model Comparison

* Seven different classical machine learning models (including Logistic Regression, SVMs, and tree-based ensembles) are trained on the data using a `TfidfVectorizer`.
* The performance of each model is evaluated using F1-score and accuracy.
* A grouped bar chart is generated to visually compare the results, allowing for a clear identification of the top-performing baseline models.

### 4. Phase 3: Hyperparameter Tuning (GridSearchCV)

* The top-performing models from the baseline comparison undergo an exhaustive hyperparameter search using `GridSearchCV`.
* This step systematically tests different combinations of settings to find the optimal configuration for each model, leading to the selection of a definitive "champion" classical model.

### 5. Phase 4: Deep Learning with LSTM

* The project then explores a deep learning approach by building a Bidirectional Long Short-Term Memory (LSTM) network.
* The text data is specially prepared for the neural network through tokenization and padding.
* A baseline LSTM is trained, followed by a hyperparameter search using `KerasTuner` to find its optimal architecture and learning rate.

### 6. Phase 5: State-of-the-Art Transformer Models

* To benchmark against the latest in NLP, the project implements two Transformer models: DistilBERT and the larger BERT model.
* These models are fine-tuned on the project's specific dataset, a common and effective technique for leveraging large, pre-trained language models.

### 7. Phase 6: Final Analysis and Conclusion

* A final comparison chart is generated, showing the performance of the champion models from each phase (Tuned LinearSVC, Tuned LSTM, and Fine-Tuned Transformers).
* The overall best-performing model is declared the project champion.
* An in-depth analysis of the champion model is conducted, including a **Confusion Matrix** to visualize its errors and an extraction of the **Top Predictive Keywords** to understand its decision-making process.
* The final trained model is saved to a file for future use.
* The learning curves of the deep learning models are visualized to show their training progress.
