# README: Text Classification of News Articles

## Overview
This project demonstrates a machine learning pipeline for classifying news articles into predefined categories such as **Tech**, **Business**, **Politics**, **Sports**, and **Entertainment**. By leveraging natural language processing (NLP) techniques and various machine learning models, the project automates the categorization of articles based on their content. The pipeline includes data preprocessing, feature engineering, model training, and evaluation to ensure accurate predictions.

## Motivation
Manual categorization of news articles is labor-intensive and prone to inconsistencies. Automating this task with machine learning models:
- Saves time.
- Enhances scalability.
- Provides a consistent and reliable classification mechanism.

This implementation is suitable for news websites, content management systems, or applications requiring automated news categorization.

---

## Key Features
1. **Text Preprocessing**: 
   - Tokenization, lemmatization, removal of stopwords, special characters, and HTML tags.
   - Conversion of text to lowercase for uniformity.

2. **Feature Engineering**:
   - Bag of Words (BoW) and TF-IDF vectorization for transforming textual data into numerical formats.

3. **Model Training**:
   - Machine learning algorithms evaluated include Logistic Regression, Random Forest, Support Vector Machines, Decision Trees, and K-Nearest Neighbors.

4. **Model Evaluation**:
   - Metrics such as accuracy, precision, recall, and F1-score are calculated to determine model performance.

5. **Prediction**:
   - Predicts the category of new articles based on input text.

---

## Dataset
The dataset, sourced from Kaggle, contains:
- **Article ID**: Unique identifier for each record.
- **Text**: Headline and article content.
- **Category**: Predefined category label for the article.

### Example Categories:
- **Tech**: News related to technology advancements, gadgets, and the tech industry.
- **Business**: Updates on the economy, markets, and corporate news.
- **Politics**: Political events, policies, and commentary.
- **Sports**: Sporting events, teams, and athletes.
- **Entertainment**: Movies, celebrities, and pop culture.

---

## Pipeline
1. **Exploratory Data Analysis (EDA)**:
   - Looked into data info and dataset to understand the distribution of categories and key words.

2. **Data Preprocessing**:
   - Removal of special characters, stopwords, and tags.
   - Tokenization and lemmatization for reducing words to their base forms.

3. **Feature Extraction**:
   - Bag of Words model to represent text numerically.
   - Vectorization for feature generation.

4. **Model Training & Evaluation**:
   - Splitting the data into training and testing sets.
   - Training models with hyperparameter tuning.
   - Evaluation using metrics like accuracy, precision, recall, and F1-score.

5. **Prediction**:
   - Model predicts the category of unseen news articles.

---

## Acknowledgments
- **Dataset**: BBC News Articles from Kaggle.
- Libraries: Scikit-learn, NLTK, Matplotlib, Seaborn, WordCloud.
