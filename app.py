from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn import metrics

import nltk
nltk.download('stopwords')

app = Flask(__name__, template_folder='templates')

def preprocessing(content):
    news_dataset = pd.read_csv(r'E:\FakeNewsPrediction\fake-news\train.csv')


    news_dataset = news_dataset.fillna('')
    news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']
    news_dataset['content'] = news_dataset['content'].apply(stemming)

    X = news_dataset['content'].values
    Y = news_dataset['label'].values

    vectorizer = TfidfVectorizer()
    vectorizer.fit(X)
    X = vectorizer.transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    model = LogisticRegression()
    model.fit(X_train, Y_train)

    return model, vectorizer

def stemming(content):
    port_stem = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

def predict_fake_news(model, vectorizer, text):
    X_new = vectorizer.transform([text])
    prediction = model.predict(X_new)

    if prediction[0] == 0:
        return 'The news is Real'
    else:
        return 'The news is Fake'

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    result = predict_fake_news(model, vectorizer, text)

    return render_template('result.html', text=text, result=result)

if __name__ == '__main__':
    model, vectorizer = preprocessing('content')
    app.static_folder = 'static'
    app.run(debug=True)
