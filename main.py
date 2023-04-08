# Осуществим предобработку данных с Твиттера, чтобы отчищенный данные в дальнейшем использовать для задачи классификации.
# Данный датасет содержит негативные (label = 1) и нейтральные (label = 0) высказывания.
# Для работы объединим train_df и test_df.
#
# Задания:#
# 1) Заменим html-сущности (к примеру: < > &). "<" заменим на “<” и "&" заменим на “&”)""".
# Сделаем это с помощью HTMLParser.unescape(). Всю предобработку делаем в новом столбце 'clean_tweet'#
# 2) Удалим @user из всех твитов с помощью паттерна "@[\w]*". Для этого создадим функцию:
# для того, чтобы найти все вхождения паттерна в тексте, необходимо использовать re.findall(pattern, input_txt)
# для для замены @user на пробел, необходимо использовать re.sub() при применении функции необходимо использовать np.vectorize(function).
# 3) Изменим регистр твитов на нижний с помощью .lower().
# 4) Заменим сокращения с апострофами (пример: ain't, can't) на пробел, используя apostrophe_dict.
# Для этого необходимо сделать функцию: для каждого слова в тексте проверить (for word in text.split()),
# если слово есть в словаре apostrophe_dict в качестве ключа (сокращенного слова),
# то заменить ключ на значение (полную версию слова).
# 5) Заменим сокращения на их полные формы, используя short_word_dict.
# Для этого воспользуемся функцией, используемой в предыдущем пункте.#
# 6) Заменим эмотиконы (пример: ":)" = "happy") на пробелы, используя emoticon_dict.
# Для этого воспользуемся функцией, используемой в предыдущем пункте.
# 7) Заменим пунктуацию на пробелы, используя re.sub() и паттерн r'[^\w\s]'.
# 8) Заменим спец. символы на пробелы, используя re.sub() и паттерн r'[^a-zA-Z0-9]'.
# 9) Заменим числа на пробелы, используя re.sub() и паттерн r'[^a-zA-Z]'.
# 10) Удалим из текста слова длиной в 1 символ, используя ' '.join([w for w in x.split() if len(w)>1]).
# 11) Поделим твиты на токены с помощью nltk.tokenize.word_tokenize, создав новый столбец 'tweet_token'.
# 12) Удалим стоп-слова из токенов, используя nltk.corpus.stopwords. Создадим столбец 'tweet_token_filtered' без стоп-слов.
# 13) Применим стемминг к токенам с помощью nltk.stem.PorterStemmer. Создадим столбец 'tweet_stemmed' после применения стемминга.
# 14) Применим лемматизацию к токенам с помощью nltk.stem.wordnet.WordNetLemmatizer.
# Создадим столбец 'tweet_lemmatized' после применения лемматизации.
# 15) Сохраним результат предобработки в pickle-файл.

import pandas as pd
from nltk import tokenize as tknz
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
from html import parser
import pickle


def html_unescape(text):
    text = parser.unescape(text)
    return text


def find_replace(text, pattern, replace_on):
    result = re.findall(pattern, text)
    for item in result:
        text = re.sub(item, replace_on, text)
    return text


def to_lower_case(text):
    return text.lower()


def replace_by_dict(text, key_value):
    for key, value in key_value:
        text = text.replace(key, value)
    return text


def user_replace(text):
    user_pattern = '@[\w]*'
    return find_replace(text, user_pattern, ' ')


def apostrophe(text):
    apostrophe_dict = {
        "ain't": "am not / are not / is not / has not / have not",
        "aren't": "are not / am not",
        "can't": "cannot",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had / he would",
        "he'll": "he shall / he will",
        "he's": "he has / he is",
        "how'd": "how did / how would",
        "how'll": "how will",
        "how's": "how has / how is",
        "i'd": "I had / I would",
        "i'll": "I shall / I will",
        "i'm": "I am",
        "i've": "I have",
        "isn't": "is not",
        "it'd": "it had / it would",
        "it'll": "it shall / it will",
        "it's": "it has / it is",
        "let's": "let us",
        "might've": "might have",
        "mightn't": "might not",
        "must've": "must have",
        "mustn't": "must not",
        "shan't": "shall not",
        "she'd": "she had / she would",
        "she'll": "she shall / she will",
        "she's": "she has / she is",
        "should've": "should have",
        "shouldn't": "should not",
        "that's": "that has / that is",
        "there's": "there has / there is",
        "they'd": "they had / they would",
        "they'll": "they shall / they will",
        "they're": "they are",
        "they've": "they have",
        "wasn't": "was not",
        "we'd": "we had / we would",
        "we'll": "we shall / we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what shall / what will",
        "what're": "what are",
        "what's": "what has / what is",
        "what've": "what have",
        "where's": "where has / where is",
        "who'd": "who had / who would",
        "who'll": "who shall / who will",
        "who's": "who has / who is",
        "who've": "who have",
        "won't": "will not",
        "would've": "would have",
        "wouldn't": "would not",
        "you'd": "you had / you would",
        "you'll": "you shall / you will",
        "you're": "you are",
        "you've": "you have"
    }

    return replace_by_dict(text, apostrophe_dict.items())


def short_words(text):
    short_word_dict = {
        "121": "one to one",
        "a/s/l": "age, sex, location",
        "adn": "any day now",
        "afaik": "as far as I know",
        "afk": "away from keyboard",
        "aight": "alright",
        "alol": "actually laughing out loud",
        "b4": "before",
        "b4n": "bye for now",
        "bak": "back at the keyboard",
        "bf": "boyfriend",
        "bff": "best friends forever",
        "bfn": "bye for now",
        "bg": "big grin",
        "bta": "but then again",
        "btw": "by the way",
        "cid": "crying in disgrace",
        "cnp": "continued in my next post",
        "cp": "chat post",
        "cu": "see you",
        "cul": "see you later",
        "cul8r": "see you later",
        "cya": "bye",
        "cyo": "see you online",
        "dbau": "doing business as usual",
        "fud": "fear, uncertainty, and doubt",
        "fwiw": "for what it's worth",
        "fyi": "for your information",
        "g": "grin",
        "g2g": "got to go",
        "ga": "go ahead",
        "gal": "get a life",
        "gf": "girlfriend",
        "gfn": "gone for now",
        "gmbo": "giggling my butt off",
        "gmta": "great minds think alike",
        "h8": "hate",
        "hagn": "have a good night",
        "hdop": "help delete online predators",
        "hhis": "hanging head in shame",
        "iac": "in any case",
        "ianal": "I am not a lawyer",
        "ic": "I see",
        "idk": "I don't know",
        "imao": "in my arrogant opinion",
        "imnsho": "in my not so humble opinion",
        "imo": "in my opinion",
        "iow": "in other words",
        "ipn": "I’m posting naked",
        "irl": "in real life",
        "jk": "just kidding",
        "l8r": "later",
        "ld": "later, dude",
        "ldr": "long distance relationship",
        "llta": "lots and lots of thunderous applause",
        "lmao": "laugh my ass off",
        "lmirl": "let's meet in real life",
        "lol": "laugh out loud",
        "ltr": "longterm relationship",
        "lulab": "love you like a brother",
        "lulas": "love you like a sister",
        "luv": "love",
        "m/f": "male or female",
        "m8": "mate",
        "milf": "mother I would like to fuck",
        "oll": "online love",
        "omg": "oh my god",
        "otoh": "on the other hand",
        "pir": "parent in room",
        "ppl": "people",
        "r": "are",
        "rofl": "roll on the floor laughing",
        "rpg": "role playing games",
        "ru": "are you",
        "shid": "slaps head in disgust",
        "somy": "sick of me yet",
        "sot": "short of time",
        "thanx": "thanks",
        "thx": "thanks",
        "ttyl": "talk to you later",
        "u": "you",
        "ur": "you are",
        "uw": "you’re welcome",
        "wb": "welcome back",
        "wfm": "works for me",
        "wibni": "wouldn't it be nice if",
        "wtf": "what the fuck",
        "wtg": "way to go",
        "wtgp": "want to go private",
        "ym": "young man",
        "gr8": "great"
    }
    return replace_by_dict(text, short_word_dict.items())


def emoticons(text):
    emoticon_dict = {
        ":)": "happy",
        ":‑)": "happy",
        ":-]": "happy",
        ":-3": "happy",
        ":->": "happy",
        "8-)": "happy",
        ":-}": "happy",
        ":o)": "happy",
        ":c)": "happy",
        ":^)": "happy",
        "=]": "happy",
        "=)": "happy",
        "<3": "happy",
        ":-(": "sad",
        ":(": "sad",
        ":c": "sad",
        ":<": "sad",
        ":[": "sad",
        ">:[": "sad",
        ":{": "sad",
        ">:(": "sad",
        ":-c": "sad",
        ":-< ": "sad",
        ":-[": "sad",
        ":-||": "sad"
    }
    return replace_by_dict(text, emoticon_dict.items())


def punctuation(text):
    pattern = '\[^\w\s\]'
    return find_replace(text, pattern, ' ')


def special_characters(text):
    pattern = '\[^a-zA-Z0-9\]'
    return find_replace(text, pattern, ' ')


def numbers_remove(text):
    pattern = '\[^a-zA-Z\]'
    return find_replace(text, pattern, ' ')


def one_symbol_remove(text):
    return ' '.join([w for w in text.split() if len(w) > 1])


def transform_row(x):
    x = html_unescape(x)
    x = user_replace(x)
    x = to_lower_case(x)
    x = apostrophe(x)
    x = short_words(x)
    x = emoticons(x)
    x = punctuation(x)
    x = special_characters(x)
    x = numbers_remove(x)
    x = one_symbol_remove(x)
    return x


def tokenize_data(text):
    return tknz.word_tokenize(text)


def stop_words(tokens):
    stop_words_english = set(stopwords.words("english"))
    return [word for word in tokens if not word in stop_words_english]


def stemming(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]


def lemmatizing(tokens):
    wn_lematizer = WordNetLemmatizer()
    return [wn_lematizer.lemmatize(word) for word in tokens]


def make_transform(data_f):
    data_f['clean_tweet'] = data_f['tweet'].apply(transform_row)
    data_f['tweet_token'] = data_f['clean_tweet'].apply(tokenize_data)
    data_f['tweet_token_filtered'] = data_f['tweet_token'].apply(stop_words)
    data_f['tweet_stemmed'] = data_f['tweet_token_filtered'].apply(stemming)
    data_f['tweet_lemmatized'] = data_f['tweet_stemmed'].apply(lemmatizing)
    return data_f


def combine_data():
    df_train = pd.read_csv("train_tweets.csv")
    df_test = pd.read_csv("test_tweets.csv")
    df_complex = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    return df_complex


def perform_words():
    df = combine_data()
    print(df.head(3))
    df = make_transform(df)
    print(df.head(3))
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump(df, f)


if __name__ == '__main__':
    perform_words()
