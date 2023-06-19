import re
import string
import csv
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option("max_rows", None)
import collections
import pickle
import matplotlib.pyplot as plt
import nltk.corpus
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import mean_squared_error
from pathlib import Path

import random
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUT_PATH = Path("results")

tfidf_dict = collections.defaultdict(list)

from tf_idf import tf_idf_results
from word2vec import w2v_results
from glove import glove_results
from fastt import fastt
from roberta import transformers_model
from final_prediction import predict


# create a sample of 100k (in the way explained in the report) and split it to train and test
def load_sample_data():
    # load sample of size 100k from the data
    num_rows = 300000
    df = pd.read_csv('data_chunk_2.csv', nrows=num_rows)
    df2 = pd.read_csv('data_chunk_1.csv', nrows=num_rows)
    print("loaded labeled")
    df = df.append(df2, ignore_index=True)
    df = df[df['founded'].notnull()]
    most_df = df[:100000]
    groups = most_df.groupby('industry')

    train_df = pd.DataFrame(columns=most_df.columns)
    test_df = pd.DataFrame(columns=most_df.columns)
    indices = groups.indices

    for index, industry in enumerate(indices.keys()):
        print(index)
        if len(indices[industry]) > 9:
            x = 5
            for i in range(len(indices[industry])):
                added_row = most_df.iloc[indices[industry][i]]
                if i < 0.8 * len(indices[industry]):
                    train_df = train_df.append(added_row)
                else:
                    test_df = test_df.append(added_row)
        else:
            print(industry)
            print(len(indices[industry]))
            tmp_df = df[df['industry'] == industry]
            for j in range(tmp_df.shape[0]):
                added_row = tmp_df.iloc[j]
                if j < 0.8 * tmp_df.shape[0]:
                    train_df = train_df.append(added_row)
                else:
                    test_df = test_df.append(added_row)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    all_df = train_df.append(test_df, ignore_index=True)

    with open('train_df_100k.pkl', 'wb') as f:
        pickle.dump(train_df, f)
    with open('test_df_100k.pkl', 'wb') as f:
        pickle.dump(test_df, f)
    with open('all_df.pkl', 'wb') as f:
        pickle.dump(all_df, f)


def manhattan_distance(a, b):
    return np.abs(a - b).sum()


def text_preprocessing(text):
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
    return lemmatized_output


# add a column of approximate size of the company
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_df = pd.read_pickle('train_df_100k.pkl')
    train_df.drop(['url', 'website', 'linkedin'], axis=1, inplace=True)
    test_df = pd.read_pickle('test_df_100k.pkl')
    test_df.drop(['url', 'website', 'linkedin'], axis=1, inplace=True)

    # calculate results for the baseline
    avg_per_industry = train_df.groupby('industry')['founded'].mean()
    real_y = list(test_df.founded)
    pred_y = []
    for idx, row in test_df.iterrows():
        pred = avg_per_industry[row['industry']]
        pred_y.append(round(pred))
    baseline_mse = mean_squared_error(real_y, pred_y)
    baseline_l1 = manhattan_distance(np.array(real_y, dtype=int), np.array(pred_y, dtype=int))
    print(f'The MSE for the baseline is: {round(baseline_mse, 3)}')
    print(f'The L1 for the baseline is: {np.round(baseline_l1, 3)}')
    print()

    # apply pre processing on the text and some pre processing on our data
    train_df.text = train_df.text.astype(str)
    test_df.text = test_df.text.astype(str)
    train_df['processed_text'] = train_df.text.apply(text_preprocessing)
    test_df['processed_text'] = test_df.text.apply(text_preprocessing)


    train_df["combined_text"] = " The industry is " + train_df["industry"] + ". The company name is " + train_df['company_name'] \
                                + ". The location of the company is " + train_df['country'] + " " + train_df['region'] \
                                + " " + train_df['locality'] + ". The information is " + train_df["text"]
    test_df["combined_text"] = " The industry is " + test_df["industry"] + ". The company name is " + test_df['company_name'] \
                               + ". The location of the company is " + test_df['country'] + " " + test_df['region'] +\
                               " " + test_df['locality'] + ". The information is " + test_df["text"]
    train_df["combined_clean_text"] = " The industry is " + train_df["industry"] + ". The company name is " + train_df['company_name'] \
                                      + ". The location of the company is " + train_df['country'] + " " + train_df['region'] \
                                      + " " + train_df['locality'] + ". The information is " + train_df["processed_text"]
    test_df["combined_clean_text"] = " The industry is " + test_df["industry"] + ". The company name is " + test_df['company_name'] \
                                     + ". The location of the company is " + test_df['country'] + " " + test_df['region'] \
                                     + " " + test_df['locality'] + ". The information is " + test_df["processed_text"]

    train_df = approx_size(train_df)
    test_df = approx_size(test_df)

    train_df["roberta_text"] = " The industry is " + train_df["industry"] + ". The company name is " + train_df['company_name'] \
                               + ". The approximate size is " + train_df['approx_size'].astype(str) + ". The location of the company is " + \
                               train_df['country'] + " " + train_df['region'] + " " + train_df['locality'] + ". The information is " + train_df["text"]
    test_df["roberta_text"] = " The industry is " + test_df["industry"] + ". The company name is " + test_df['company_name'] \
                              + ". The approximate size is " + test_df['approx_size'].astype(str) + ". The location of the company is " + \
                              test_df['country'] + " " + test_df['region'] + " " + test_df['locality'] + ". The information is " + test_df["text"]


    train_df.to_pickle('ready_train_100k.pkl')
    test_df.to_pickle('ready_test_100k.pkl')

    # train_df = pd.read_pickle('ready_train_100k.pkl')
    # test_df = pd.read_pickle('ready_test_100k.pkl')

    # calls to all representations (with models) that were used
    tf_idf_results(train_df, test_df)

    w2v_results(train_df, test_df)
    glove_results(train_df, test_df)

    fastt(train_df, test_df)

    transformers_model('roberta_text')

    # # I created the graph manually because results were gotten individually for every representation
    # global_results_dict = {'baseline': 708.579, 'tf-idf': 605.131, 'Word2Vec': 620.818,
    #                        'glove': 648.268, 'fasttext': 1095.082, 'RoBerta': 803.802}
    # keys = global_results_dict.keys()
    # values = global_results_dict.values()
    # plt.bar(keys, values)
    # plt.title('Best result for each representation')
    # plt.xlabel('Representations')
    # plt.ylabel('MSE')
    # plt.show()

    # call to the file that make the prediction on the missing data
    predict(train_df, test_df)
