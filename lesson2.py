# 1. Создайте мешок слов с помощью
# sklearn.feature_extraction.text.CountVectorizer.fit_transform(). Применим его к 'tweet_stemmed'
# и 'tweet_lemmatized' отдельно.
# ● Игнорируем слова, частота которых в документе строго превышает порог 0.9 с
# помощью max_df.
# ● Ограничим количество слов, попадающий в мешок, с помощью max_features =
# 1000.
# ● Исключим стоп-слова с помощью stop_words='english'.
# ● Отобразим Bag-of-Words модель как DataFrame. columns необходимо извлечь с
# помощью CountVectorizer.get_feature_names().
# 2. Создайте мешок слов с помощью
# sklearn.feature_extraction.text.TfidfVectorizer.fit_transform(). Применим его к 'tweet_stemmed' и
# 'tweet_lemmatized' отдельно.
# ● Игнорируем слова, частота которых в документе строго превышает порог 0.9 с
# помощью max_df.
# ● Ограничим количество слов, попадающий в мешок, с помощью max_features =
# 1000.
# ● Исключим стоп-слова с помощью stop_words='english'.
# ● Отобразим Bag-of-Words модель как DataFrame. columns необходимо извлечь с
# помощью TfidfVectorizer.get_feature_names().
# 3. Проверьте ваши векторайзеры на корпусе который использовали на вебинаре, составьте
# таблицу метод векторизации и скор который вы получили (в методах векторизации по
# изменяйте параметры что бы добиться лучшего скора) обратите внимание как
# падает/растёт скор при уменьшении количества фичей, и изменении параметров, так же
# попробуйте применить к векторайзерам PCA для сокращения размерности посмотрите на
# качество сделайте выводы

import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn import model_selection, preprocessing, linear_model
from sklearn.metrics import accuracy_score


def read_data_from_pkl(filename):
    with open(filename, 'rb') as f:
        x = pickle.load(f)
    return x


df_init = read_data_from_pkl('processed_data.pkl')
cv = CountVectorizer(analyzer=lambda x: x, max_df=0.9, max_features=1000)

bow_stemmed = cv.fit_transform(df_init['tweet_stemmed'])
col_names = cv.vocabulary_
df = pd.DataFrame(bow_stemmed.toarray(), columns=[word for word in col_names])
print(df.head())

tf_idf_vec = TfidfVectorizer(analyzer=lambda x: x, max_df=0.9, max_features=1000)
X = tf_idf_vec.fit_transform(df_init['tweet_stemmed'])
col_names = tf_idf_vec.vocabulary_
df_tf_idf = pd.DataFrame(X.toarray(), columns=[word for word in col_names])
print(df_tf_idf.head())

# Загружаем данные из корпуса
data = open('corpus').read()
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split()
    labels.append(content[0])
    texts.append(" ".join(content[1:]))

# создаем df
trainDF = pd.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels
print(trainDF.head(5))

train_x, valid_x, train_y, valid_y = train_test_split(trainDF['text'], trainDF['label'])

# labelEncode целевую переменную
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])

xtrain_count = count_vect.transform(train_x)
xvalid_count = count_vect.transform(valid_x)

classifier = linear_model.LogisticRegression()
classifier.fit(xtrain_count, train_y)
predictions = classifier.predict(xvalid_count)
print(predictions)

print(accuracy_score(valid_y, predictions))

