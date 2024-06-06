# PHASE_3_PROJECT
Churn Prediction Project
Overview
This project focuses on predicting customer churn using various machine learning models. The dataset contains customer information, and the target variable indicates whether a customer has churned.

Project Structure
Data Preprocessing
Handling missing values
Encoding categorical variables
Scaling numerical features
Exploratory Data Analysis (EDA)
Visualizing feature distributions
Correlation analysis
Model Training and Evaluation
Training Logistic Regression, K-Nearest Neighbors (KNN), and Decision Tree models
Evaluating model performance using accuracy, precision, recall, F1-score, and ROC-AUC
Model Interpretation and Deployment
Analyzing feature importance
Saving and loading the model for future use
Installation
To run this project, you need to have Python installed along with the following libraries:

pandas
numpy
scikit-learn
seaborn
matplotlib
joblib
You can install the required libraries using:

bash
Copy code
pip install pandas numpy scikit-learn seaborn matplotlib joblib
Usage
Preprocessing: Prepare the data by handling missing values, encoding categorical variables, and scaling numerical features.
EDA: Perform exploratory data analysis to understand the data distribution and relationships.
Model Training: Train different machine learning models and evaluate their performance.
Model Evaluation: Use evaluation metrics to compare the models and select the best one.
Model Interpretation: Analyze feature importance and interpret the model.
Deployment: Save the trained model and deploy it using a web framework like Flask.
Results
The best model achieved the following performance:

Accuracy: 0.85
Precision: 0.78
Recall: 0.75
F1-score: 0.76
ROC-AUC: 0.82
Conclusion
This project demonstrates the process of building a churn prediction model from data preprocessing to model deployment. The Decision Tree model provided a good balance between accuracy and interpretability.