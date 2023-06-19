import pandas as pd
import sys
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option("max_rows", None)
import collections
import matplotlib.pyplot as plt

import nltk.corpus
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor

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


def manhattan_distance(a, b):
    return np.abs(a - b).sum()


def linear_regression_tfidf(X_train, y_train, X_test, y_test, status, name_of_column):
    reg = LinearRegression(normalize=True, n_jobs=-1)
    if status == 'sum':
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
    reg.fit(X_train, y_train)
    reg_pred = reg.predict(np.array(X_test))
    reg_pred = list(map(int, reg_pred))
    y_test = list(map(int, y_test))
    tfidf_mse = mean_squared_error(y_test, reg_pred)
    score = round(reg.score(X_test, y_test), 3)
    l1_dist = manhattan_distance(np.array(y_test, dtype=int), np.array(reg_pred, dtype=int))

    comb = 'linear regression on ' + name_of_column + ' using ' + status
    tfidf_dict['linear regression'].append((comb, round(tfidf_mse, 3)))
    if status == 'sum':
        print(f'    The MSE for linear regression on tfidf (using sum) is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for linear regression on tfidf (using sum) is: {round(l1_dist, 3)}')
        print(f'    The linear regression score for tfidf (using sum) is: {score}')
    elif status == 'sum with size':
        print(f'    The MSE for linear regression on tfidf (using sum) and size is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for linear regression on tfidf (using sum) and size is: {round(l1_dist, 3)}')
        print(f'    The linear regression score for tfidf (using sum) and size is: {score}')
    elif status == 'not sum':
        print(f'    The MSE for linear regression on tfidf (without aggregation) is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for linear regression on tfidf (without aggregation) is: {round(l1_dist, 3)}')
        print(f'    The linear regression score for tfidf (without aggregation) is: {score}')
    else:
        print(f'    The MSE for linear regression on tfidf (without aggregation) and size is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for linear regression on tfidf (without aggregation) and size is: {round(l1_dist, 3)}')
        print(f'    The linear regression score for tfidf (without aggregation) and size is: {score}')


def DecisionTreeRegressor_tfidf(X_train, y_train, X_test, y_test, status, name_of_column):
    reg = DecisionTreeRegressor(random_state=42, max_depth=3)
    if status == 'sum':
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
    reg.fit(X_train, y_train)
    reg_pred = reg.predict(np.array(X_test))
    reg_pred = list(map(int, reg_pred))
    y_test = list(map(int, y_test))
    tfidf_mse = mean_squared_error(y_test, reg_pred)
    score = round(reg.score(X_test, y_test), 3)
    l1_dist = manhattan_distance(np.array(y_test, dtype=int), np.array(reg_pred, dtype=int))


    comb = 'DecisionTreeRegressor on ' + name_of_column + ' using ' + status
    tfidf_dict['DecisionTreeRegressor'].append((comb, round(tfidf_mse, 3)))
    if status == 'sum':
        print(f'    The MSE for DecisionTreeRegressor on tfidf (using sum) is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for DecisionTreeRegressor on tfidf (using sum) is: {round(l1_dist, 3)}')
        print(f'    The DecisionTreeRegressor score for tfidf (using sum) is: {score}')
    elif status == 'sum with size':
        print(f'    The MSE for DecisionTreeRegressor on tfidf (using sum) and size is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for DecisionTreeRegressor on tfidf (using sum) and size is: {round(l1_dist, 3)}')
        print(f'    The DecisionTreeRegressor score for tfidf (using sum) and size is: {score}')
    elif status == 'not sum':
        print(f'    The MSE for DecisionTreeRegressor on tfidf (without aggregation) is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for DecisionTreeRegressor on tfidf (without aggregation) is: {round(l1_dist, 3)}')
        print(f'    The DecisionTreeRegressor score for tfidf (without aggregation) is: {score}')
    else:
        print(f'    The MSE for DecisionTreeRegressor on tfidf (without aggregation) and size is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for DecisionTreeRegressor on tfidf (without aggregation) and size is: {round(l1_dist, 3)}')
        print(f'    The DecisionTreeRegressor score for tfidf (without aggregation) and size is: {score}')


def linSVC_tfidf(X_train, y_train, X_test, y_test, status, name_of_column):
    if status == 'sum':
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
    linSVC = make_pipeline(StandardScaler(with_mean=False),
                        LinearSVC(dual=False, class_weight='balanced', random_state=42, max_iter=5000)).fit(X_train, y_train)
    linSVC_pred = linSVC.predict(np.array(X_test))
    linSVC_pred = list(map(int, linSVC_pred))
    y_test = list(map(int, y_test))
    tfidf_mse = mean_squared_error(y_test, linSVC_pred)
    accuracy_score_linSVC = round(linSVC.score(X_test, y_test), 3)
    l1_dist = manhattan_distance(np.array(y_test, dtype=int), np.array(linSVC_pred, dtype=int))

    comb = 'linear SVC on ' + name_of_column + ' using ' + status
    tfidf_dict['linear SVC'].append((comb, round(tfidf_mse, 3)))
    if status == 'sum':
        print(f'    The MSE for linSVC on tfidf (using sum) is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for linSVC on tfidf (using sum) is: {round(l1_dist, 3)}')
        print(f'    The linSVC score for tfidf (using sum) is: {accuracy_score_linSVC}')
    elif status == 'sum with size':
        print(f'    The MSE for linSVC on tfidf (using sum) and size is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for linSVC on tfidf (using sum) and size is: {round(l1_dist, 3)}')
        print(f'    The linSVC score for tfidf (using sum) and size is: {accuracy_score_linSVC}')
    elif status == 'not sum':
        print(f'    The MSE for linSVC on tfidf (without aggregation) is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for linSVC on tfidf (without aggregation) is: {round(l1_dist, 3)}')
        print(f'    The linSVC score for tfidf (without aggregation) is: {accuracy_score_linSVC}')
    else:
        print(f'    The MSE for linSVC on tfidf (without aggregation) and size is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for linSVC on tfidf (without aggregation) and size is: {round(l1_dist, 3)}')
        print(f'    The linSVC score for tfidf (without aggregation) and size is: {accuracy_score_linSVC}')


def logistic_regression_tfidf(X_train, y_train, X_test, y_test, status, name_of_column):
    if status == 'sum':
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
    log_reg = make_pipeline(StandardScaler(with_mean=False),
                        LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced', n_jobs=-1)).fit(X_train, y_train)
    log_reg_pred = log_reg.predict(np.array(X_test))
    log_reg_pred = list(map(int, log_reg_pred))
    y_test = list(map(int, y_test))
    tfidf_mse = mean_squared_error(y_test, log_reg_pred)
    accuracy_score_log_reg = round(log_reg.score(X_test, y_test), 3)
    l1_dist = manhattan_distance(np.array(y_test, dtype=int), np.array(log_reg_pred, dtype=int))

    comb = 'logistic regression on ' + name_of_column + ' using ' + status
    tfidf_dict['logistic regression'].append((comb, round(tfidf_mse, 3)))
    if status == 'sum':
        print(f'    The MSE for log_reg on tfidf (using sum) is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for log_reg on tfidf (using sum) is: {round(l1_dist, 3)}')
        print(f'    The log_reg score for tfidf (using sum) is: {accuracy_score_log_reg}')
    elif status == 'sum with size':
        print(f'    The MSE for log_reg on tfidf (using sum) and size is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for log_reg on tfidf (using sum) and size is: {round(l1_dist, 3)}')
        print(f'    The log_reg score for tfidf (using sum) and size is: {accuracy_score_log_reg}')
    elif status == 'not sum':
        print(f'    The MSE for log_reg on tfidf (without aggregation) is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for log_reg on tfidf (without aggregation) is: {round(l1_dist, 3)}')
        print(f'    The log_reg score for tfidf (without aggregation) is: {accuracy_score_log_reg}')
    else:
        print(f'    The MSE for log_reg on tfidf (without aggregation) and size is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for log_reg on tfidf (without aggregation) and size is: {round(l1_dist, 3)}')
        print(f'    The log_reg score for tfidf (without aggregation) and size is: {accuracy_score_log_reg}')


def KNN_tfidf(X_train, y_train, X_test, y_test, status, name_of_column):
    if status == 'sum':
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
    knn = KNeighborsClassifier(n_neighbors=15, weights='distance', metric='minkowski', p=1, n_jobs=-1).fit(X_train, y_train)
    knn_pred = knn.predict(np.array(X_test))
    knn_pred = list(map(int, knn_pred))
    y_test = list(map(int, y_test))
    tfidf_mse = mean_squared_error(y_test, knn_pred)
    accuracy_score_knn = round(knn.score(X_test, y_test), 3)
    l1_dist = manhattan_distance(np.array(y_test, dtype=int), np.array(knn_pred, dtype=int))

    comb = 'KNN on ' + name_of_column + ' using ' + status
    tfidf_dict['KNN'].append((comb, round(tfidf_mse, 3)))
    if status == 'sum':
        print(f'    The MSE for KNN on tfidf (using sum) is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for KNN on tfidf (using sum) is: {round(l1_dist, 3)}')
        print(f'    The KNN score for tfidf (using sum) is: {accuracy_score_knn}')
    elif status == 'sum with size':
        print(f'    The MSE for KNN on tfidf (using sum) and size is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for KNN on tfidf (using sum) and size is: {round(l1_dist, 3)}')
        print(f'    The KNN score for tfidf (using sum) and size is: {accuracy_score_knn}')
    elif status == 'not sum':
        print(f'    The MSE for KNN on tfidf (without aggregation) is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for KNN on tfidf (without aggregation) is: {round(l1_dist, 3)}')
        print(f'    The KNN score for tfidf (without aggregation) is: {accuracy_score_knn}')
    else:
        print(f'    The MSE for KNN on tfidf (without aggregation) and size is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for KNN on tfidf (without aggregation) and size is: {round(l1_dist, 3)}')
        print(f'    The KNN score for tfidf (without aggregation) and size is: {accuracy_score_knn}')


def MLP_tfidf(X_train, y_train, X_test, y_test, status, name_of_column):
    if status == 'sum':
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
    mlp = MLPClassifier(alpha=0.01, random_state=42, max_iter=1000, learning_rate='adaptive').fit(X_train, y_train)
    mlp_pred = mlp.predict(np.array(X_test))
    mlp_pred = list(map(int, mlp_pred))
    y_test = list(map(int, y_test))
    tfidf_mse = mean_squared_error(y_test, mlp_pred)
    score = round(mlp.score(X_test, y_test), 3)
    l1_dist = manhattan_distance(np.array(y_test, dtype=int), np.array(mlp_pred, dtype=int))

    comb = 'MLP on ' + name_of_column + ' using ' + status
    tfidf_dict['MLP'].append((comb, round(tfidf_mse, 3)))
    if status == 'sum':
        print(f'    The MSE for MLP on tfidf (using sum) is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for MLP on tfidf (using sum) is: {round(l1_dist, 3)}')
        print(f'    The MLP score for tfidf (using sum) is: {score}')
    elif status == 'sum with size':
        print(f'    The MSE for MLP on tfidf (using sum) and size is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for MLP on tfidf (using sum) and size is: {round(l1_dist, 3)}')
        print(f'    The MLP score for tfidf (using sum) and size is: {score}')
    elif status == 'not sum':
        print(f'    The MSE for MLP on tfidf (without aggregation) is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for MLP on tfidf (without aggregation) is: {round(l1_dist, 3)}')
        print(f'    The MLP score for tfidf (without aggregation) is: {score}')
    else:
        print(f'    The MSE for MLP on tfidf (without aggregation) and size is: {round(tfidf_mse, 3)}')
        print(f'    The L1 for MLP on tfidf (without aggregation) and size is: {round(l1_dist, 3)}')
        print(f'    The MLP score for tfidf (without aggregation) and size is: {score}')


def tfidf(train_df, test_df, name_of_column):
    v = TfidfVectorizer(max_features=500)
    tmp = pd.concat([train_df[name_of_column], test_df[name_of_column]])
    x = v.fit_transform(tmp)

    df1 = pd.DataFrame(x.toarray(), columns=v.get_feature_names()).head(len(train_df))
    df1['Sum'] = df1.sum(axis=1)
    df1_sum = df1['Sum'].to_numpy()
    df2 = pd.DataFrame(x.toarray(), columns=v.get_feature_names()).tail(len(test_df))
    df2['Sum'] = df2.sum(axis=1)
    df2_sum = df2['Sum'].to_numpy()

    df1_size = pd.concat([df1, train_df['approx_size']], axis=1)
    df2_size = df2.copy()
    df2_size['approx_size'] = test_df['approx_size'].to_numpy()

    df1_sum_size = pd.concat([df1['Sum'], train_df['approx_size']], axis=1)
    df2_sum_size = np.stack((df2_sum, test_df['approx_size'].to_numpy()), axis=-1)

    if name_of_column == 'text':
        print('The prints below are calculated on not pre-processed text:')
        print()
    elif name_of_column == 'processed_text':
        print('The prints below are calculated on pre-processed text:')
        print()
    elif name_of_column == 'combined_clean_text':
        print('The prints below are calculated on pre-processed text, combined with other columns:')
        print()
    else:
        print('The prints below are calculated on not pre-processed text, combined with other columns:')
        print()

    linear_regression_tfidf(df1, train_df['founded'], df2, test_df['founded'], 'not sum', name_of_column)

    linear_regression_tfidf(df1_sum, train_df['founded'], df2_sum, test_df['founded'], 'sum', name_of_column)

    linear_regression_tfidf(df1_size, train_df['founded'], df2_size, test_df['founded'], 'not sum with size', name_of_column)

    linear_regression_tfidf(df1_sum_size, train_df['founded'], df2_sum_size, test_df['founded'], 'sum with size', name_of_column)
    print()
    # linSVC_tfidf(df1, train_df['founded'], df2, test_df['founded'], 'not sum', name_of_column)
    #
    # linSVC_tfidf(df1_sum, train_df['founded'], df2_sum, test_df['founded'], 'sum', name_of_column)
    #
    # linSVC_tfidf(df1_size, train_df['founded'], df2_size, test_df['founded'], 'not sum with size', name_of_column)
    #
    # linSVC_tfidf(df1_sum_size, train_df['founded'], df2_sum_size, test_df['founded'], 'sum with size', name_of_column)
    # print()
    # logistic_regression_tfidf(df1, train_df['founded'], df2, test_df['founded'], 'not sum', name_of_column)
    #
    # logistic_regression_tfidf(df1_sum, train_df['founded'], df2_sum, test_df['founded'], 'sum', name_of_column)
    #
    # logistic_regression_tfidf(df1_size, train_df['founded'], df2_size, test_df['founded'], 'not sum with size', name_of_column)
    #
    # logistic_regression_tfidf(df1_sum_size, train_df['founded'], df2_sum_size, test_df['founded'], 'sum with size', name_of_column)
    # print()
    KNN_tfidf(df1, train_df['founded'], df2, test_df['founded'], 'not sum', name_of_column)

    KNN_tfidf(df1_sum, train_df['founded'], df2_sum, test_df['founded'], 'sum', name_of_column)

    KNN_tfidf(df1_size, train_df['founded'], df2_size, test_df['founded'], 'not sum with size', name_of_column)

    KNN_tfidf(df1_sum_size, train_df['founded'], df2_sum_size, test_df['founded'], 'sum with size', name_of_column)
    print()
    MLP_tfidf(df1, train_df['founded'], df2, test_df['founded'], 'not sum', name_of_column)

    MLP_tfidf(df1_sum, train_df['founded'], df2_sum, test_df['founded'], 'sum', name_of_column)

    MLP_tfidf(df1_size, train_df['founded'], df2_size, test_df['founded'], 'not sum with size', name_of_column)

    MLP_tfidf(df1_sum_size, train_df['founded'], df2_sum_size, test_df['founded'], 'sum with size', name_of_column)
    print()
    DecisionTreeRegressor_tfidf(df1, train_df['founded'], df2, test_df['founded'], 'not sum', name_of_column)

    DecisionTreeRegressor_tfidf(df1_sum, train_df['founded'], df2_sum, test_df['founded'], 'sum', name_of_column)

    DecisionTreeRegressor_tfidf(df1_size, train_df['founded'], df2_size, test_df['founded'], 'not sum with size', name_of_column)

    DecisionTreeRegressor_tfidf(df1_sum_size, train_df['founded'], df2_sum_size, test_df['founded'], 'sum with size', name_of_column)


def tf_idf_results(train_df, test_df):
    tfidf(train_df, test_df, 'text')
    print()
    tfidf(train_df, test_df, 'processed_text')
    print()
    tfidf(train_df, test_df, 'combined_text')
    print()
    tfidf(train_df, test_df, 'combined_clean_text')

    best_dict = dict()
    for key, val in tfidf_dict.items():
        min_mse = sys.float_info.max
        best_tpl = val[0]
        for tpl in val:
            if tpl[1] < min_mse:
                min_mse = tpl[1]
                best_tpl = tpl
            else:
                continue
        best_dict[key] = best_tpl
    print(best_dict)

    keys = []
    values = []
    for key, val in best_dict.items():
        keys.append(key)
        values.append(val[1])
    plt.bar(keys, values)
    plt.title('Models results on tf-idf')
    plt.xlabel('Models')
    plt.ylabel('MSE')
    plt.show()
