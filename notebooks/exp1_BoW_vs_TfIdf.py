import mlflow
import pandas as pd
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = {
    'bow': CountVectorizer(),
    'tfidf': CountVectorizer()
}
algos = {
    'logistic_regression': LogisticRegression(max_iter=1000),
    'MultinomialNB': MultinomialNB(),
    'RandomForestClassifier': RandomForestClassifier(), 
    'SVC': SVC(),
    'GradientBoostingClassifier': GradientBoostingClassifier()
}

import dagshub
dagshub.init(repo_owner='Sudip-8345', repo_name='DVC-git-mini-Project', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/Sudip-8345/DVC-git-mini-Project.mlflow')

mlflow.set_experiment("tweet_emotion_classification_exp_v2")

vectorizers = {'bow': CountVectorizer(), 'tfidf': TfidfVectorizer()}

algos = {
    'logistic_regression': LogisticRegression(max_iter=1000),
    'MultinomialNB': MultinomialNB(),
    'RandomForestClassifier': RandomForestClassifier(), 
    'SVC': SVC(),
    'GradientBoostingClassifier': GradientBoostingClassifier()
}

for algo_name, algo in algos.items():
    for vectorizer_name, vectorizer in vectorizers.items():
        with mlflow.start_run():
            X = vectorizer.fit_transform(df['content'])
            y = df['sentiment']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            algo.fit(X_train, y_train)
            y_pred = algo.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            mlflow.log_param("vectorizer", vectorizer_name)
            mlflow.log_param("algorithm", algo_name)
            mlflow.log_param('test_size', 0.2)

            if algo_name == 'logistic_regression':
                mlflow.log_param("max_iter", algo.max_iter)
            elif algo_name == 'SVC':
                mlflow.log_param("kernel", algo.kernel)
            elif algo_name == 'RandomForestClassifier':
                mlflow.log_param("n_estimators", algo.n_estimators)
                mlflow.log_param("max_depth", algo.max_depth)
            elif algo_name == 'GradientBoostingClassifier':
                mlflow.log_param("n_estimators", algo.n_estimators)
                mlflow.log_param("learning_rate", algo.learning_rate)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("f1_score", f1)

            # Optional: log model artifact
            # import joblib
            # joblib.dump(algo, "model.pkl")
            # mlflow.log_artifact("model.pkl")

            print(f"Algorithm: {algo_name}, Vectorizer: {vectorizer_name}, Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1 Score: {f1}")

            # model_dir = "exp1_baseline_model"
            # import joblib
            # joblib.dump(model, "model/logreg_model.pkl")
            # mlflow.log_artifact("model/logreg_model.pkl", artifact_path="model")
