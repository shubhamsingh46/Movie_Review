
# A) Importing necessary libraries and modules
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# B) Loading the dataset and preprocessing using NLTK
df = pd.read_csv("IMDB_Dataset.csv")
df.head()

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)
df['review'] = df['review'].apply(clean_text)


# C) Feature extraction using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['review']).toarray()
y = df['sentiment'].map({'positive':1, 'negative':0})

# D) Splitting the dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# E) Training and evaluating the models
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# F) Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# G) Support Vector Machine
svm = LinearSVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# H) Evaluating the models
def evaluate(y_test, y_pred):
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

lr_results = evaluate(y_test, y_pred_lr)
nb_results = evaluate(y_test, y_pred_nb)
svm_results = evaluate(y_test, y_pred_svm)

results = pd.DataFrame([lr_results, nb_results, svm_results], index=['Logistic Regression', 'Naive Bayes', 'SVM'])
print(results)

# I) Saving the best model and vectorizer using pickle
import pickle
pickle.dump(svm, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
