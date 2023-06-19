import pandas as pd
import numpy as np
import collections
import re
import string
import nltk.corpus
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

tfidf_dict = collections.defaultdict(list)

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


def approx_size(df):
    approx_size_list = []
    for idx, row in df.iterrows():
        if row['size'] == '1-10':
            approx_size_list.append(5)
        elif row['size'] == '11-50':
            approx_size_list.append(30)
        elif row['size'] == '51-200':
            approx_size_list.append(125)
        elif row['size'] == '201-500':
            approx_size_list.append(350)
        elif row['size'] == '501-1000':
            approx_size_list.append(750)
        elif row['size'] == '1001-5000':
            approx_size_list.append(3000)
        elif row['size'] == '5001-10000':
            approx_size_list.append(7500)
        else:
            approx_size_list.append(10000)
    df['approx_size'] = approx_size_list
    return df


def manhattan_distance(a, b):
    return np.abs(a - b).sum()


def text_preprocessing(text):
    if isinstance(text, str):
        text = text.lower()
        text = text.encode('ascii', 'ignore').decode()
        text = ' '.join([word for word in text.split(' ') if word not in stop_words])
        text = re.sub("@\S+", " ", text)
        text = re.sub("https*\S+", " ", text)
        text = re.sub("#\S+", " ", text)
        text = re.sub("\'\w+", '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = re.sub(r'\w*\d+\w*', '', text)
        text = re.sub('\s{2,}', " ", text)
        word_list = nltk.word_tokenize(text)
        lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    else:
        lemmatized_output = text
    return lemmatized_output


def creating_df_for_prediction(): #taking rows with no foundation year and also transforming to tf_idf before predicting
    df = pd.read_csv('data_chunk_0.csv')
    df2 = pd.read_csv('data_chunk_1.csv')
    df3 = pd.read_csv('data_chunk_2.csv')
    df = df.append(df2, ignore_index=True)
    df = df.append(df3, ignore_index=True)
    predicting_df = df[df['founded'].isnull()]
    predicting_df['processed_text'] = predicting_df.text.apply(text_preprocessing)
    predicting_df["combined_clean_text"] = " The industry is " + predicting_df["industry"] + ". The company name is " +\
                                           predicting_df['company_name'] + ". The location of the company is " + predicting_df['country'] \
                                           + " " +predicting_df['region'] + " " + predicting_df['locality']\
                                           + ". The information is " + predicting_df["processed_text"]
    predicting_df = approx_size(predicting_df)
    combined_clean_text = predicting_df['combined_clean_text']
    combined_clean_text.at[95452, 0] = ''
    combined_clean_text.at[221456, 0] = ''
    combined_clean_text.at[1004030, 0] = ''
    combined_clean_text.at[1040836, 0] = ''
    combined_clean_text.at[1219966, 0] = ''

    v = TfidfVectorizer(max_features=500)
    x = v.fit_transform(combined_clean_text)
    predicting_tfidf_df = pd.DataFrame(x.toarray(), columns=v.get_feature_names())
    predicting_tfidf_df['Sum'] = predicting_tfidf_df.sum(axis=1)
    df_size = predicting_df['approx_size'].reset_index()
    predicting_tfidf_df = pd.concat([predicting_tfidf_df, df_size['approx_size']], axis=1)
    company_websites = predicting_df['website']
    return company_websites,predicting_tfidf_df


def linear_regression_tfidf(X_train, y_train, X_test, y_test, status, name_of_column):
    reg = LinearRegression(normalize=True, n_jobs=-1)
    reg.fit(X_train, y_train)
    return reg


def tfidf(train_df, test_df, name_of_column):
    v = TfidfVectorizer(max_features=500)
    tmp = pd.concat([train_df[name_of_column], test_df[name_of_column]])
    x = v.fit_transform(tmp)

    df1 = pd.DataFrame(x.toarray(), columns=v.get_feature_names()).head(len(train_df))
    df1['Sum'] = df1.sum(axis=1)
    df2 = pd.DataFrame(x.toarray(), columns=v.get_feature_names()).tail(len(test_df))
    df2['Sum'] = df2.sum(axis=1)

    df1_size = pd.concat([df1, train_df['approx_size']], axis=1)
    df2_size = df2.copy()
    df2_size['approx_size'] = test_df['approx_size'].to_numpy()

    model_reg = linear_regression_tfidf(df1_size, train_df['founded'], df2_size, test_df['founded'], 'not sum with size', name_of_column)
    return model_reg


def predict(train_df, test_df):
    model_reg = tfidf(train_df, test_df, 'combined_clean_text')
    company_website, predicting_tfidf_df = creating_df_for_prediction()
    new_foundation_year = pd.DataFrame(model_reg. predict(predicting_tfidf_df))
    founded_year_int = [int(i) for i in list(new_foundation_year[0])]
    for index,year in enumerate(founded_year_int):
        if year > 2021:
            founded_year_int[index] = 2021

    new_foundation_year_df = pd.DataFrame({'company_name': list(company_website),'founded': founded_year_int})
    new_foundation_year_df.to_csv('foundation_year_prediction.csv', index=False, header=['website','founded'])
