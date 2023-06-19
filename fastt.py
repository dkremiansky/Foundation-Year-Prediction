import re
import string
import zipfile
import gc
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option("max_rows", None)
import pickle

import nltk.corpus
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer

from pathlib import Path

import random
import torch
import fasttext

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

"""#**Global functions for all models** """

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


def fasttext_preprocess(train_df, test_df): # continue of preprocess for fasttext
    train_df["text_with_size"] = ". The approximate size is " + train_df['approx_size'].astype(str) + ". The information is " + train_df["text"]

    test_df["text_with_size"] = ". The approximate size is " +  test_df['approx_size'].astype(str) + ". The information is " + test_df["text"]

    train_df['processed_text_with_size'] = ". The approximate size is " + train_df['approx_size'].astype(str) + ". The information is " + train_df["processed_text"]
    test_df['processed_text_with_size'] = ". The approximate size is " +  test_df['approx_size'].astype(str) + ". The information is " + test_df["processed_text"]

    train_df["combined_clean_text_with_size"] = " The industry is " + train_df["industry"] + ". The company name is " + train_df['company_name'] \
                             + ". The approximate size is " + train_df['approx_size'].astype(str) + ". The location of the company is " + \
                             train_df['country'] + " " + train_df['region'] + " " + train_df['locality'] + ". The information is " + train_df["processed_text"]
    test_df["combined_clean_text_with_size"] = " The industry is " + test_df["industry"] + ". The company name is " + test_df['company_name'] \
                            + ". The approximate size is " +  test_df['approx_size'].astype(str) + ". The location of the company is " + \
                             test_df['country'] + " " + test_df['region'] + " " + test_df['locality'] + ". The information is " + test_df["processed_text"]


    fastt_train_df = train_df.copy()
    fastt_train_df[['founded','approx_size']] = fastt_train_df[['founded','approx_size']].astype(str)
    fastt_train_df['labeled year'] = fastt_train_df['founded'].apply(lambda x: '__label__' + x)

    fastt_test_df = test_df.copy()
    fastt_test_df[['founded','approx_size']] = fastt_test_df[['founded','approx_size']].astype(str)
    fastt_test_df['labeled year'] = fastt_test_df['founded'].apply(lambda x: '__label__' + x)

    return fastt_train_df, fastt_test_df



def fastt_pred_to_int(pred_str):
    pred_str = re.sub('[%s]' % re.escape(string.punctuation), ' ', pred_str)
    pred = [int(s) for s in pred_str.split() if s.isdigit()]
    return pred[0]


def fasttext_vizualization(text_mse, text_mse_with_size, processed_text_mse, processed_text_mse_with_size,
                           combined_clean_text_mse, combined_clean_text_size_mse, combined_text_mse,
                           combined_text_size_mse):
    fasttext_results_dict = {'not pre-processed': text_mse_with_size, 'pre-processed': processed_text_mse_with_size,
                             'pre-processed combined': combined_clean_text_size_mse,
                             'not pre-processed combined': combined_text_size_mse}
    keys = fasttext_results_dict.keys()
    values = fasttext_results_dict.values()
    plt.bar(keys, values)
    plt.title('FastText results')
    plt.xlabel('Text with size combinations')
    plt.ylabel('MSE')
    plt.show()

    fasttext_results_dict = {'not pre-processed': text_mse, 'pre-processed': processed_text_mse,
                             'pre-processed combined': combined_clean_text_mse,
                             'not pre-processed combined': combined_text_mse}
    keys = fasttext_results_dict.keys()
    values = fasttext_results_dict.values()
    plt.bar(keys, values)
    plt.title('FastText results')
    plt.xlabel('Text without size combinations')
    plt.ylabel('MSE')
    plt.show()

def fastt_pred(train_df, test_df, train_txt, test_txt, name_of_column): #returning list of predictions

  # Training the fastText classifier
  model = fasttext.train_supervised(train_txt, epoch=10, dim=50, wordNgrams = 2, loss='hs')
  # Evaluating performance on the entire test file
  model.test(test_txt)

  # Save trained model
  model.save_model('fasttext_model.bin')

  pred_list = []
  for i in range(len(test_df)):
    model_pred = model.predict(test_df[name_of_column][i])
    pred_str = model_pred[0][0]
    pred_list.append(fastt_pred_to_int(pred_str))

  return pred_list


def fastt(train_df, test_df):
    fastt_train_df, fastt_test_df = fasttext_preprocess(train_df, test_df)
    # Saving the CSV file as a text file to train/test the classifier
    fastt_train_df[['labeled year', 'text']].to_csv('fastt_train.txt',
                                                    index=False,
                                                    sep=' ',
                                                    header=None,
                                                    quoting=csv.QUOTE_NONE,
                                                    quotechar="",
                                                    escapechar=" ")
    fastt_test_df[['labeled year', 'text']].to_csv('fastt_test.txt',
                                                   index=False,
                                                   sep=' ',
                                                   header=None,
                                                   quoting=csv.QUOTE_NONE,
                                                   quotechar="",
                                                   escapechar=" ")

    text_pred_list = fastt_pred(fastt_train_df, fastt_test_df, 'fastt_train.txt', 'fastt_test.txt', 'text')
    text_mse = mean_squared_error(fastt_test_df['founded'].to_list(), text_pred_list)
    print(f'    MSE for fasttext calculated on not pre-processed text without size is: {round(text_mse, 3)}')

    text_size_pred_list = fastt_pred(fastt_train_df, fastt_test_df, 'fastt_train.txt', 'fastt_test.txt', 'text_with_size')
    text_mse_with_size = mean_squared_error(fastt_test_df['founded'].to_list(), text_size_pred_list)
    print(f'    MSE for fasttext calculated on not pre-processed text with size is: {round(text_mse_with_size, 3)}')

    processed_text_pred_list = fastt_pred(fastt_train_df, fastt_test_df, 'fastt_train.txt', 'fastt_test.txt', 'processed_text')
    processed_text_mse = mean_squared_error(fastt_test_df['founded'].to_list(), processed_text_pred_list)
    print(f'    MSE for fasttext calculated on pre-processed text without size is: {round(processed_text_mse, 3)}')

    processed_text_size_pred_list = fastt_pred(fastt_train_df, fastt_test_df, 'fastt_train.txt', 'fastt_test.txt', 'processed_text_with_size')
    processed_text_mse_with_size = mean_squared_error(fastt_test_df['founded'].to_list(), processed_text_size_pred_list)
    print(f'    MSE for fasttext calculated on pre-processed text with size is: {round(processed_text_mse_with_size, 3)}')

    combined_clean_text_pred_list = fastt_pred(fastt_train_df, fastt_test_df, 'fastt_train.txt', 'fastt_test.txt', 'combined_clean_text')
    combined_clean_text_mse = mean_squared_error(fastt_test_df['founded'].to_list(), combined_clean_text_pred_list)
    print(f'    MSE for fasttext calculated on pre-processed text without size, combined with other columns is: {round(combined_clean_text_mse, 3)}')

    combined_clean_text_size_pred_list = fastt_pred(fastt_train_df, fastt_test_df, 'fastt_train.txt', 'fastt_test.txt', 'combined_clean_text_with_size')
    combined_clean_text_size_mse = mean_squared_error(fastt_test_df['founded'].to_list(), combined_clean_text_size_pred_list)
    print(f'    MSE for fasttext calculated on pre-processed text with size, combined with other columns is: {round(combined_clean_text_size_mse, 3)}')

    combined_text_pred_list = fastt_pred(fastt_train_df, fastt_test_df, 'fastt_train.txt', 'fastt_test.txt', 'combined_text')
    combined_text_mse = mean_squared_error(fastt_test_df['founded'].to_list(), combined_text_pred_list)
    print(f'    MSE for fasttext calculated on not pre-processed text without size, combined with other columns is: {round(combined_text_mse, 3)}')

    combined_text_size_pred_list = fastt_pred(fastt_train_df, fastt_test_df, 'fastt_train.txt', 'fastt_test.txt', 'roberta_text')
    combined_text_size_mse = mean_squared_error(fastt_test_df['founded'].to_list(), combined_text_size_pred_list)
    print(f'    MSE for fasttext calculated on not pre-processed text with size, combined with other columns is: {round(combined_text_size_mse, 3)}')

    fasttext_vizualization(text_mse, text_mse_with_size, processed_text_mse, processed_text_mse_with_size,
                           combined_clean_text_mse, combined_clean_text_size_mse, combined_text_mse,
                           combined_text_size_mse)
