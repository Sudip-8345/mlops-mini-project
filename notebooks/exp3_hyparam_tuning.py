import mlflow
import pandas as pd
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv').drop(columns=['tweet_id'])
df.head()

def remove_punctuation(text):
    return re.sub(f"[{re.escape(string.punctuation)}]", "", text)

def lowercase(text):
    text = [word.lower() for word in text.split()]
    return " ".join(text)

def remove_stopwords(text):
    stopwords_list = set(stopwords.words('english'))
    for word in text.split():
        if word.lower() in stopwords_list:
            text = text.replace(word, "")
    return text

def remove_nums(text):
    for char in text:
        if char.isdigit():
            text = text.replace(char, "")
    return text

def lematize(text):
    lemmatizer = WordNetLemmatizer()
    for word in text.split():
        text = text.replace(word, lemmatizer.lemmatize(word))
    return text

def remove_urls(text):
    return re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

def preprocess_text(text):
    text = remove_punctuation(text)
    text = lowercase(text)
    text = remove_stopwords(text)
    text = remove_nums(text)
    text = lematize(text)
    text = remove_urls(text)
    return text

def normalize_text(df):
    try:
        df['content'] = df['content'].astype(str)
        df['content'] = df['content'].apply(preprocess_text)
        return df
    except Exception as e:
        print(f"Error in normalize_text: {e}")
        raise

df = df[df['sentiment'].isin(['happiness', 'sadness'])]
df = normalize_text(df)

df.sentiment = df.sentiment.map({
    # 'neutral': 0,
    # 'worry': 1,
    'sadness': 0,
    'happiness':1
})
# df.head()
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale'],
    # 'degree': [2, 3, 4]
}

import dagshub
dagshub.init(repo_owner='Sudip-8345', repo_name='DVC-git-mini-Project', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/Sudip-8345/DVC-git-mini-Project.mlflow')

mlflow.set_experiment("SVM Hyperparameter Tuning with TfIdf Vectorization")

from sklearn.model_selection import GridSearchCV
vectorizer = TfidfVectorizer()
with mlflow.start_run():
    X = vectorizer.fit_transform(df['content'])
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    grid_search = GridSearchCV(SVC(), param_grid, cv=3, scoring='f1',n_jobs=-1,verbose=1)
    grid_search.fit(X_train, y_train)

    for params,mean_score, std_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['std_test_score']):
        with mlflow.start_run(nested=True,run_name=f"SVM with params={params}"):
            model = SVC(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

            print(f"Params: {params}, Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1 Score: {f1}")
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    for key, value in best_params.items():
        mlflow.log_param(f"best_{key}", value)

    mlflow.log_metric("best_score", best_score)

    # Evaluate best model
    y_pred_best = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_best)
    recall = recall_score(y_test, y_pred_best)
    precision = precision_score(y_test, y_pred_best)
    f1 = f1_score(y_test, y_pred_best)

    mlflow.log_metric("best_accuracy", accuracy)
    mlflow.log_metric("best_recall", recall)
    mlflow.log_metric("best_precision", precision)
    mlflow.log_metric("best_f1_score", f1)

    print(f"Best Params: {best_params}, Best Score: {best_score}")
    print(f"Final Model Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1 Score: {f1}")

    # mlflow.sklearn.log_model(best_model, "model")

    # import joblib
    # joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    # mlflow.log_artifact("tfidf_vectorizer.pkl")
