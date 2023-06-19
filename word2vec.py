import pandas as pd
import numpy as np
import pickle
pd.set_option('display.max_columns', None)
pd.set_option("max_rows", None)
from matplotlib import pyplot as plt

from gensim.models import Word2Vec

import nltk.corpus

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor
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


def manhattan_distance(a, b):
    return np.abs(a - b).sum()


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key


def linear_regression_w2v(X_train, y_train, X_test, y_test, status):
    reg = LinearRegression(normalize=True, n_jobs=-1)
    reg.fit(X_train, y_train)
    reg_pred = reg.predict(np.array(X_test))
    reg_pred = list(map(int, reg_pred))
    y_test = list(map(int, y_test))
    lin_reg_mse = mean_squared_error(y_test, reg_pred)
    score = round(reg.score(X_test, y_test), 3)
    manhattan_dist_lin_reg = manhattan_distance(np.array(y_test, dtype=int), np.array(reg_pred, dtype=int))
    print(f'    The results for linear regression with status', status, 'are :')
    print(f'        The MSE is: {round(lin_reg_mse, 3)}')
    print(f'       The score is: {score}')
    print(f'        The manhattan distance is: {manhattan_dist_lin_reg}')
    print()
    return lin_reg_mse


def DecisionTreeRegressor_w2v(X_train, y_train, X_test, y_test, status):
    min_mse = float('inf')
    depth_for_min_mse = 0  # initiation
    for max_depth in range(2, 6):
        reg = DecisionTreeRegressor(random_state=42, max_depth=max_depth)
        reg.fit(X_train, y_train)
        reg_pred = reg.predict(np.array(X_test))
        reg_pred = list(map(int, reg_pred))
        y_test = list(map(int, y_test))
        decision_tree_mse = mean_squared_error(y_test, reg_pred)
        score = round(reg.score(X_test, y_test), 3)
        manhattan_dist_decision_tree = manhattan_distance(np.array(y_test, dtype=int), np.array(reg_pred, dtype=int))
        print(f'    The results for desicion tree with depth = ', max_depth, 'with status', status, 'are :')
        print(f'        The MSE is: {round(decision_tree_mse, 3)}')
        print(f'       The decision tree score is: {score}')
        print(f'        The manhattan distance is: {manhattan_dist_decision_tree}')
        print()
        if decision_tree_mse < min_mse:
            min_mse = decision_tree_mse
            depth_for_min_mse = max_depth
    return min_mse, depth_for_min_mse


def linSVC_w2v(X_train, y_train, X_test, y_test, status):
    linSVC = make_pipeline(StandardScaler(with_mean=False),
                           LinearSVC(dual=False, class_weight='balanced', random_state=42, max_iter=5000)).fit(X_train,
                                                                                                               y_train)
    linSVC_pred = linSVC.predict(X_test)
    linSVC_pred = list(map(int, linSVC_pred))
    y_test = list(map(int, y_test))
    score_linSVC = round(linSVC.score(y_test, linSVC_pred), 3)
    linSVC_mse = mean_squared_error(y_test, linSVC_pred)
    manhattan_dist_linSVC = manhattan_distance(np.array(y_test, dtype=int), np.array(linSVC_pred, dtype=int))
    print(f'    The results for linSVC with status', status, 'are :')
    print(f'        The MSE is: {round(linSVC_mse, 3)}')
    print(f'       The linSVC score is: {score_linSVC}')
    print(f'        The manhattan distance is: {manhattan_dist_linSVC}')
    print()
    return linSVC_mse


def logistic_regression_w2v(X_train, y_train, X_test, y_test, status):
    log_reg = make_pipeline(StandardScaler(with_mean=False),
                            LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced', n_jobs=-1)).fit(
        X_train, y_train)
    log_reg_pred = log_reg.predict(X_test)
    log_reg_pred = list(map(int, log_reg_pred))
    y_test = list(map(int, y_test))
    score_log_reg = round(log_reg.score(y_test, log_reg_pred), 3)
    log_reg_mse = mean_squared_error(y_test, log_reg_pred)
    manhattan_dist_log_reg = manhattan_distance(np.array(y_test, dtype=int), np.array(log_reg_pred, dtype=int))
    print(f'    The results for logistic regression with status', status, 'are :')
    print(f'        The MSE is: {round(log_reg_mse, 3)}')
    print(f'       The linSVC score is: {score_log_reg}')
    print(f'        The manhattan distance is: {manhattan_dist_log_reg}')
    print()
    return log_reg_mse


def KNN_w2v(X_train, y_train, X_test, y_test, status):
    best_k = 0  # initiation
    min_mse = float('inf')
    for k in range(11, 31):
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='minkowski', p=1, n_jobs=-1).fit(X_train,
                                                                                                              y_train)
        knn_pred = knn.predict(X_test)
        knn_pred = list(map(int, knn_pred))
        y_test = list(map(int, y_test))
        score_knn = round(knn.score(X_test, y_test), 3)
        knn_mse = mean_squared_error(y_test, knn_pred)
        manhattan_dist_knn = manhattan_distance(np.array(y_test, dtype=int), np.array(knn_pred, dtype=int))
        print(f'    The results for knn with k = ', k, 'with status', status, 'are :')
        print(f'        The MSE is: {round(knn_mse, 3)}')
        print(f'       The KNN score is: {score_knn}')
        print(f'        The manhattan distance is: {manhattan_dist_knn}')
        print()
        if knn_mse < min_mse:
            min_mse = knn_mse
            best_k = k
    return min_mse, best_k


def MLP_w2v(X_train, y_train, X_test, y_test, status):
    best_alpha = 0  # initiation
    min_mse = float('inf')
    list_alpha = [0.005, 0.01, 0.02]
    for alpha in list_alpha:
        mlp = MLPClassifier(alpha=alpha, random_state=42, max_iter=1000, learning_rate='adaptive').fit(X_train, y_train)
        mlp_pred = mlp.predict(X_test)
        mlp_pred = list(map(int, mlp_pred))
        y_test = list(map(int, y_test))
        mlp_mse = mean_squared_error(y_test, mlp_pred)
        score_mlp = round(mlp.score(X_test, y_test), 3)
        manhattan_dist_mlp = manhattan_distance(np.array(y_test, dtype=int), np.array(mlp_pred, dtype=int))
        print(f'    The results for mlp with alpha = ', alpha, 'with status', status, 'are :')
        print(f'        The MSE is: {round(mlp_mse, 3)}')
        print(f'      The linSVC score is: {score_mlp}')
        print(f'        The manhattan distance is: {manhattan_dist_mlp}')
        print()
        if mlp_mse < min_mse:
            min_mse = mlp_mse
            best_alpha = alpha
    return min_mse, best_alpha


def w2v(train_df, test_df, name_of_column):
    full_df = pd.concat([train_df, test_df])
    full_approx_size_list = list(full_df['approx_size'])
    sentences = list(full_df[name_of_column])
    for index, sentence in enumerate(sentences):
        sentences[index] = sentence.split()
    print("sentences split")
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4, epochs=100)
    word_vectors = model.wv
    print("word2vec done")
    list_vectors_sum = []
    list_vectors_avg = []
    for sentence in sentences:
        list_sentence_vectors = []
        for word in sentence:
            word_vector = list(word_vectors.vectors[word_vectors.key_to_index[word]])
            list_sentence_vectors.append(word_vector)
        if len(list_sentence_vectors) != 0:
            sentence_vector_sum = [sum(x) for x in zip(*list_sentence_vectors)]
            sentence_vector_avg = [x / len(sentence) for x in sentence_vector_sum]
        else:
            sentence_vector_sum = list([0] * 100)
            sentence_vector_avg = list([0] * 100)
        list_vectors_sum.append(sentence_vector_sum)
        list_vectors_avg.append(sentence_vector_avg)
    print("vectors are done")

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

    train_labels = list(full_df[:len(train_df)]['founded'])
    test_labels = list(full_df[len(train_df):]['founded'])
    mse_dict = {}

    # sum without size

    train_vectors = list_vectors_sum[:len(train_df)]
    test_vectors = list_vectors_sum[len(train_df):]

    lin_reg_sum_without_size = linear_regression_w2v(train_vectors, train_labels, test_vectors, test_labels,'sum without size')
    decision_tree_sum_without_size, best_depth = DecisionTreeRegressor_w2v(train_vectors, train_labels, test_vectors,test_labels, 'sum without size')
    knn_sum_without_size, best_k = KNN_w2v(train_vectors, train_labels, test_vectors, test_labels, 'sum without size')
    mlp_sum_without_size, best_alpha = MLP_w2v(train_vectors, train_labels, test_vectors, test_labels,'sum without size')
    # linSVC_w2v(train_vectors, train_labels, test_vectors, test_labels,'sum without size')
    # logistic_regression_w2v(train_vectors, train_labels, test_vectors, test_labels, 'sum without size')

    mse_dict['lin_reg_sum_without_size'] = lin_reg_sum_without_size
    mse_dict[f'decision_tree_sum_without_size with depth = {best_depth}'] = decision_tree_sum_without_size
    mse_dict[f'KNN_sum_without_size with k =  {best_k}'] = knn_sum_without_size
    mse_dict[f'MLP_sum_without_size with alpha = {best_alpha}'] = mlp_sum_without_size

    # sum with size

    for index, vector in enumerate(list_vectors_sum):
        list_vectors_sum[index].append(full_approx_size_list[index])
    train_vectors = list_vectors_sum[:len(train_df)]
    test_vectors = list_vectors_sum[len(train_df):]

    lin_reg_sum_with_size = linear_regression_w2v(train_vectors, train_labels, test_vectors, test_labels,'sum with size')
    decision_tree_sum_with_size, best_depth = DecisionTreeRegressor_w2v(train_vectors, train_labels, test_vectors,test_labels, 'sum with size')
    knn_sum_with_size, best_k = KNN_w2v(train_vectors, train_labels, test_vectors, test_labels, 'sum with size')
    mlp_sum_with_size, best_alpha = MLP_w2v(train_vectors, train_labels, test_vectors, test_labels, 'sum with size')
    # linSVC_w2v(train_vectors, train_labels, test_vectors, test_labels, 'sum with size')
    # logistic_regression_w2v(train_vectors, train_labels, test_vectors, test_labels, 'sum with size')

    mse_dict['lin_reg_sum_with_size'] = lin_reg_sum_with_size
    mse_dict[f'decision_tree_sum_with_size with depth = {best_depth}'] = decision_tree_sum_with_size
    mse_dict[f'KNN_sum_with_size with k =  {best_k}'] = knn_sum_with_size
    mse_dict[f'MLP_sum_with_size with alpha = {best_alpha}'] = mlp_sum_with_size

    # average without size

    train_vectors = list_vectors_avg[:len(train_df)]
    test_vectors = list_vectors_avg[len(train_df):]

    lin_reg_average_without_size = linear_regression_w2v(train_vectors, train_labels, test_vectors, test_labels, 'average without size')
    decision_tree_average_without_size, best_depth = DecisionTreeRegressor_w2v(train_vectors, train_labels, test_vectors, test_labels, 'average without size')
    knn_average_without_size, best_k = KNN_w2v(train_vectors, train_labels, test_vectors, test_labels, 'average without size')
    mlp_average_without_size, best_alpha = MLP_w2v(train_vectors, train_labels, test_vectors, test_labels, 'average without size')
    # linSVC_w2v(train_vectors, train_labels, test_vectors, test_labels, 'average without size')
    # logistic_regression_w2v(train_vectors, train_labels, test_vectors, test_labels, 'average without size')

    mse_dict['lin_reg_average_without_size'] = lin_reg_average_without_size
    mse_dict[f'decision_tree_average_without_size with depth = {best_depth}'] = decision_tree_average_without_size
    mse_dict[f'KNN_average_without_size with k =  {best_k}'] = knn_average_without_size
    mse_dict[f'MLP_average_without_size with alpha = {best_alpha}'] = mlp_average_without_size

    # average with size

    for index, vector in enumerate(list_vectors_avg):
        list_vectors_avg[index].append(full_approx_size_list[index])
    train_vectors = list_vectors_avg[:len(train_df)]
    test_vectors = list_vectors_avg[len(train_df):]

    lin_reg_average_with_size = linear_regression_w2v(train_vectors, train_labels, test_vectors, test_labels,'average with size')
    decision_tree_average_with_size, best_width = DecisionTreeRegressor_w2v(train_vectors, train_labels, test_vectors, test_labels, 'average with size')
    knn_average_with_size, best_k = KNN_w2v(train_vectors, train_labels, test_vectors, test_labels, 'average with size')
    mlp_average_with_size, best_alpha = MLP_w2v(train_vectors, train_labels, test_vectors, test_labels,'average with size')
    # linSVC_w2v(train_vectors, train_labels, test_vectors, test_labels, 'average with size')
    # logistic_regression_w2v(train_vectors, train_labels, test_vectors, test_labels, 'average with size')

    mse_dict['lin_reg_average_with_size'] = lin_reg_average_with_size
    mse_dict[f'decision_tree_average_with_size with depth = {best_depth}'] = decision_tree_average_with_size
    mse_dict[f'KNN_average_with_size with k =  {best_k}'] = knn_average_with_size
    mse_dict[f'MLP_average_with_size with alpha = {best_alpha}'] = mlp_average_with_size

    best_mse = {}
    all_values = list(mse_dict.values())
    best_lin_reg_value = min(all_values[0], all_values[4], all_values[8], all_values[12])
    best_mse[get_key(best_lin_reg_value, mse_dict)] = best_lin_reg_value
    best_decision_tree_value = min(all_values[1], all_values[5], all_values[9], all_values[13])
    best_mse[get_key(best_decision_tree_value, mse_dict)] = best_decision_tree_value
    best_knn_value = min(all_values[2], all_values[6], all_values[10], all_values[14])
    best_mse[get_key(best_knn_value, mse_dict)] = best_knn_value
    best_mlp_value = min(all_values[3], all_values[7], all_values[11], all_values[15])
    best_mse[get_key(best_mlp_value, mse_dict)] = best_mlp_value

    return best_mse


def w2v_results(train_df, test_df):
    print("Word2Vec results:")
    print()
    word2vec_text = w2v(train_df, test_df, 'text')
    word2vec_processed_text = w2v(train_df, test_df, 'processed_text')
    word2vec_combined_text = w2v(train_df, test_df, 'combined_text')
    word2vec_combined_clean_text = w2v(train_df, test_df, 'combined_clean_text')

    word2vec_keys = list(word2vec_text.keys())
    word2vec_values = list(word2vec_text.values())
    for index, key in enumerate(word2vec_keys):
        word2vec_keys[index] = key + ' on text'

    word2vec_processed_keys = list(word2vec_processed_text.keys())
    word2vec_processed_values = list(word2vec_processed_text.values())
    for index, key in enumerate(word2vec_processed_keys):
        word2vec_processed_keys[index] = key + ' on preprocessed text'

    word2vec_combined_keys = list(word2vec_combined_text.keys())
    word2vec_combined_values = list(word2vec_combined_text.values())
    for index, key in enumerate(word2vec_combined_keys):
        word2vec_combined_keys[index] = key + ' on combined text'

    word2vec_combined_clean_keys = list(word2vec_combined_clean_text.keys())
    word2vec_combined_clean_values = list(word2vec_combined_clean_text.values())
    for index, key in enumerate(word2vec_combined_clean_keys):
        word2vec_combined_clean_keys[index] = key + ' on combined clean text'

    word2vec_keys = word2vec_keys + word2vec_processed_keys + word2vec_combined_keys + word2vec_combined_clean_keys
    word2vec_values = word2vec_values + word2vec_processed_values + word2vec_combined_values + word2vec_combined_clean_values
    word2vec_dict = dict(zip(word2vec_keys, word2vec_values))

    best_word2vec = {}
    best_lin_reg_value = min(word2vec_values[0], word2vec_values[4], word2vec_values[8], word2vec_values[12])
    best_word2vec[get_key(best_lin_reg_value, word2vec_dict)] = best_lin_reg_value
    best_decision_tree_value = min(word2vec_values[1], word2vec_values[5], word2vec_values[9], word2vec_values[13])
    best_word2vec[get_key(best_decision_tree_value, word2vec_dict)] = best_decision_tree_value
    best_knn_value = min(word2vec_values[2], word2vec_values[6], word2vec_values[10], word2vec_values[14])
    best_word2vec[get_key(best_knn_value, word2vec_dict)] = best_knn_value
    best_mlp_value = min(word2vec_values[3], word2vec_values[7], word2vec_values[11], word2vec_values[15])
    best_word2vec[get_key(best_mlp_value, word2vec_dict)] = best_mlp_value

    word2vec_graph_dict = {'linear regression': best_lin_reg_value, 'KNN': best_knn_value, 'MLP': best_mlp_value,
                           'DecisionTreeRegressor': best_decision_tree_value}
    keys = word2vec_graph_dict.keys()
    values = word2vec_graph_dict.values()
    plt.bar(keys, values)
    plt.ylabel('MSE')
    plt.xlabel('Models')
    plt.title('Models results on word2vec')
    plt.show()
