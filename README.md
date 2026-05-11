# Fake News Classification using Machine Learning

This project focuses on detecting fake news articles using Machine Learning and Natural Language Processing (NLP) techniques. The project compares the performance of two classification algorithms:

- Logistic Regression (LR)
- Random Forest Classifier (RFC)

The implementation is inspired by the research paper *"Fake News Classifier with Deep Learning"* but with a modification in the preprocessing stage. Instead of using regex-based filtering as mentioned in the paper, stemming was used for text normalization.

---

# Dataset

Dataset used: ISOT Fake News Dataset from Kaggle

Dataset Link:  
https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

The dataset consists of two CSV files:

- `Fake.csv`
- `True.csv`

---

# Project Workflow

## 1. Importing Dependencies

Required libraries were imported for:
- Data manipulation
- NLP preprocessing
- Machine Learning
- Model evaluation

Libraries used include:
- pandas
- numpy
- nltk
- sklearn

---

## 2. Stopword Removal

English stopwords were removed using the NLTK stopwords corpus to eliminate unnecessary words that do not contribute significantly to prediction.

---

## 3. Data Preprocessing

The dataset originally contained two separate files:
- Fake news articles
- Real news articles

The following preprocessing steps were performed:

- Added a new column named `label`
  - `1` → Fake News
  - `0` → Real News
- Combined both datasets
- Shuffled the dataset using:

```python
frac = 1
random_state = 42
```

This ensured randomness and reproducibility.

---

## 4. Feature Engineering

The `title` and `text` columns were combined into a single column named:

```python
content
```

This combined textual information was used as the final input feature for prediction.

---

## 5. Splitting Features and Labels

The dataset was divided into:

- `X` → content
- `y` → label

---

## 6. Text Processing and Vectorization

Unlike the reference paper, this project uses **stemming** for text normalization.

Steps involved:
- Filtering and cleaning text
- Applying stemming using PorterStemmer
- Converting text into numerical vectors using:

```python
TfidfVectorizer()
```

TF-IDF helps convert textual data into meaningful numerical representations based on word importance.

---

## 7. Train-Test Split

The dataset was divided into:
- 80% Training Data
- 20% Testing Data

Using:

```python
random_state = 42
```

---

## 8. Model Training

Two machine learning models were trained:

### Logistic Regression
A linear classification model commonly used for binary classification tasks.

### Random Forest Classifier
An ensemble learning model based on multiple decision trees.

---

## 9. Model Evaluation

The models were evaluated using:
- Accuracy Score
- Classification Report

Evaluation metrics included:
- Precision
- Recall
- F1-Score

---

# Results

The Random Forest Classifier performed better compared to Logistic Regression for this dataset.

## Final Observation

- Logistic Regression achieved good baseline performance. (98.57%)
- Random Forest Classifier provided better classification capability and overall prediction performance. (99.34%)

---

# Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TF-IDF Vectorization

---

# Future Improvements

Possible future enhancements:
- Using Deep Learning models such as LSTM or BERT
- Real-time fake news detection
- Web scraping for live news verification
- Multi-language support
- Hyperparameter tuning for improved accuracy

---

# Reference Paper

This project was inspired by the following paper:

**Lu, Michael Febrianto, et al.**  
*"Fake News Classifier with Deep Learning."*  
2022 International Conference on Informatics Electrical and Electronics (ICIEE), IEEE, 2022.  
DOI: 10.1109/ICIEE55596.2022.10010120

---

# Author

Developed as a Machine Learning and NLP project for Fake News Classification and comparative model analysis.
