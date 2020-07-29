from sklearn import linear_model as lm
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import datetime
import pickle
import os
from itertools import product

plt.style.use('ggplot')


# get_data to be run before main
# get_data should take 1-2 minutes
# main may take up to 20 minutes


def time_it(func):
    def timed_func(*args, **kwargs):
        t0 = datetime.datetime.now()
        result = func(*args, **kwargs)
        t1 = datetime.datetime.now()
        print(f'Call to {func.__name__} took {(t1-t0).seconds} seconds')
        return result

    return timed_func


def get_feature_data(ticker, drop_features=(), after_year=None):
    data = pd.read_csv(f'data/feature_data/{ticker}_final_data.csv', index_col=0)
    if after_year is not None:
        start_date = datetime.datetime(after_year, 1, 1)
        data = data.loc[pd.to_datetime(data.index) >= start_date]
    data = data.drop(f'{ticker}_result', axis=1)
    features = data.columns.to_list()
    for d_ft in drop_features:
        features = [ft for ft in features if d_ft not in ft]
    return data[features]


def renormalise_data(X):
    return pd.DataFrame(preprocessing.scale(X), columns=X.columns, index=X.index)


def get_split_data(data, test_size, output_column='result_zero_one', renormalise=True):
    train_cols = data.columns.to_list()
    test_column = [c for c in train_cols if output_column in c]
    if len(test_column) != 1:
        raise Exception('Should be precisely one result column.')
    test_column = test_column[0]
    train_cols.remove(test_column)
    X, y = data[train_cols], data[test_column]
    if renormalise:
        X = renormalise_data(X)
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)


def get_classifier(clf_type, params):
    if clf_type.lower() == 'logit':
        clf = lm.LogisticRegression(random_state=RANDOM_STATE)
    elif clf_type.lower() == 'svm':
        clf = LinearSVC(random_state=RANDOM_STATE)
    elif clf_type.lower() == 'svm_':
        clf = SVC(probability=True, kernel='linear', random_state=RANDOM_STATE)
    elif clf_type.lower() == 'nb':
        clf = BernoulliNB()
    elif clf_type.lower() == 'knn':
        clf = KNeighborsClassifier()
    else:
        raise NotImplementedError(f'{clf_type.lower()} is not a supported classifier')
    clf.set_params(**params)
    return clf


@time_it
def forward_selection(X, y, clf_type, clf_params, folds=10):
    results = {}
    best_features = []
    test_features = X.columns.to_list()
    num_features = len(test_features)
    top_coef = None

    for col_num in range(num_features):
        print(col_num + 1)
        temp_results = {}
        max_score = 0

        # iterate over remaining features and fit
        for ft in test_features:
            features = best_features + [ft]
            clf = get_classifier(clf_type=clf_type, params=clf_params)
            scores = cross_val_score(clf, X[features], y, cv=folds)
            avg_score = scores.mean()
            if avg_score > max_score:
                top_coef = ft
                max_score = avg_score
            temp_results[ft] = avg_score

        # select coefficient that gave best model
        results[col_num] = max_score
        best_features += [top_coef]
        test_features.remove(top_coef)

    best_features = pd.DataFrame({'columns': best_features}, index=range(1, num_features + 1))
    results = pd.DataFrame(results.values(), index=results.keys(), columns=['accuracy'])
    return best_features, results


def run_forward_selection(tickers, test_size, RANDOM_STATE):
    for ticker in tickers:
        data = get_feature_data(ticker, drop_features=('ewma',), after_year=None)
        X_train, X_test, y_train, y_test = get_split_data(data=data, test_size=test_size,
                                                          output_column='result_zero_one')

        fwd_selection_path = f'data/results/forward_selection/{ticker}/'
        if not os.path.exists(fwd_selection_path):
            os.mkdir(fwd_selection_path)
        for clf in ['nb', 'knn', 'logit', 'svm']:
            best_coefficients, results = forward_selection(X=X_train, y=y_train, clf_type=clf,
                                                           clf_params=clf_params[clf], folds=5)
            best_coefficients.to_csv(
                f'{fwd_selection_path}{ticker}_coefs_{clf}_{round(100 * test_size)}%_R{RANDOM_STATE}.csv')
            results.to_csv(f'{fwd_selection_path}{ticker}_results_{clf}_{round(100 * test_size)}%_R{RANDOM_STATE}.csv')


@time_it
def exhaustive_feature_selection(clf_type, X, y, clf_params, folds=5):
    results = {}
    rtn_fts, mtm_fts, sma_fts, std_fts, sr_ft, im_ft, hlm_ft = get_features_names(X)
    count = 1
    for rtn_ft, mtm_ft, sma_ft, std_ft in product(rtn_fts, mtm_fts, sma_fts, std_fts):
        print(count)
        alias = f"rtn_{rtn_ft.split('_')[-1]}_mtm_{mtm_ft.split('_')[-1]}"
        alias += f"_sma_{sma_ft.split('_')[-1]}_std_{std_ft.split('_')[-1]}_sr_im_hlm"

        use_features = [rtn_ft, mtm_ft, sma_ft, std_ft, sr_ft, im_ft, hlm_ft]
        clf = get_classifier(clf_type=clf_type, params=clf_params[clf_type])
        scores = cross_val_score(clf, X[use_features], y, cv=folds)
        results[alias] = scores.mean()
        count += 1
    return pd.DataFrame(results.values(), index=results.keys(), columns=[f'{clf_type}_accuracy'])


def run_exhaustive_selection(tickers, test_size, RANDOM_STATE):
    for ticker in tickers:
        data = get_feature_data(ticker, drop_features=('ewma',), after_year=None)
        X_train, X_test, y_train, y_test = get_split_data(data=data, test_size=test_size,
                                                          output_column='result_zero_one')
        exh_selection_path = 'data/results/exhaustive_selection/'
        ticker_exh_path = f"{exh_selection_path}{ticker}_{round(100 * test_size)}%_R{RANDOM_STATE}.csv"

        exhaustion = pd.DataFrame()
        for clf_type in ['nb', 'knn', 'logit', 'svm']:
            clf_exhaustion = exhaustive_feature_selection(clf_type, X_train, y_train, clf_params=clf_params)
            exhaustion = pd.concat([exhaustion, clf_exhaustion], axis=1)
        exhaustion.to_csv(ticker_exh_path)


def get_features_names(X):
    ft_names = X.columns.to_list()
    rtn_fts = [c for c in ft_names if 'returns' in c and 'sr' not in c]
    mtm_fts = [c for c in ft_names if 'momentum' in c and 'hlm' not in c and 'im' not in c]
    sma_fts = [c for c in ft_names if 'sma' in c]
    std_fts = [c for c in ft_names if 'std' in c]
    sr_ft, im_ft, hlm_ft = ft_names[-3:]
    return rtn_fts, mtm_fts, sma_fts, std_fts, sr_ft, im_ft, hlm_ft


def create_forward_selection_analysis(tickers):
    for ticker in tickers:
        fwd_selection_path = f'data/results/forward_selection/{ticker}/'
        if not os.path.exists(fwd_selection_path):
            os.mkdir(fwd_selection_path)
        test_set = 25
        RANDOM_STATE = 39

        if not (os.path.exists(f'{fwd_selection_path}features_{test_set}%_R{RANDOM_STATE}.csv') or os.path.exists(
                f'{fwd_selection_path}results_{test_set}%_R{RANDOM_STATE}.csv')):
            features = pd.DataFrame()
            results = pd.DataFrame()
            for f in [f for f in os.listdir(fwd_selection_path) if f'R{RANDOM_STATE}' in f]:
                if f.endswith('csv'):
                    ticker, _, clf = f.split('_')[:3]
                    data = pd.read_csv(fwd_selection_path + f, index_col=0)
                    data.columns = [f'{ticker}_{clf}_{c}' for c in data.columns]
                    if 'coefs' in f:
                        features = pd.concat([features, data], axis=1)
                    elif 'results' in f:
                        results = pd.concat([results, data], axis=1)
            plt.figure(figsize=(6, 4))
            plt.plot(results, marker='x')
            plt.xlabel('Number Features')
            plt.ylabel('Accuracy')
            plt.legend(results.columns)
            plt.title('Feature Selection - Accuracy vs Features CV')
            plt.savefig(f'{fwd_selection_path}accuracy_{test_set}%_R{RANDOM_STATE}.png')
            plt.close()

            features.to_csv(f'{fwd_selection_path}features_{test_set}%_R{RANDOM_STATE}.csv')
            results.to_csv(f'{fwd_selection_path}results_{test_set}%_R{RANDOM_STATE}.csv')

        exh_selection_path = 'data/results/exhaustive_selection/'
        for f in [f for f in os.listdir(exh_selection_path) if ticker in f]:
            if f.endswith('.csv') and not 'feature' in f:
                id = f.split('.')[0]
                if not os.path.exists(f'{exh_selection_path}feature_contrb_{id}.csv'):
                    if not os.path.exists(f'{exh_selection_path}feature_contrb_{id}.csv'):
                        data = pd.read_csv(exh_selection_path + f, index_col=0)
                        plt.figure(figsize=(8, 6))
                        plt.plot(data.values)
                        plt.legend([c.split('_')[0] for c in data.columns])
                        plt.xlabel('Feature Combination Number')
                        plt.ylabel('Accuracy')
                        plt.title(f"Exhaustive Selection - {ticker}")
                        plt.savefig(f"{exh_selection_path}{id}.png")
                        plt.close()

                        split_cols = np.array([x.split('_')[:-3] for x in data.index])
                        new_cols = split_cols[0][::2]
                        data = pd.concat(
                            [data, pd.DataFrame(split_cols[:, [1, 3, 5, 7]], index=data.index, columns=new_cols)],
                            axis=1)
                        result = pd.DataFrame()
                        for nc in new_cols:
                            grouped_data = data.groupby(nc).mean()
                            temp = grouped_data.reset_index().rename(columns={f'{nc}': 'value'})
                            temp['name'] = nc + temp['value']
                            result = pd.concat([result, temp], axis=0)
                        result = result.drop('value', axis=1)
                        result.to_csv(f'{exh_selection_path}feature_contrb_{id}.csv')


def grid_search(clf_type, grid_params, X_train, y_train):
    params = {}
    if clf_type == 'logit':
        params = {'solver': 'saga'}
    clf = get_classifier(clf_type=clf_type, params=params)
    grid_clf = GridSearchCV(clf, grid_params[clf_type], cv=5)
    grid_clf.fit(X_train, y_train)
    estimated_clf = grid_clf.best_estimator_
    print(f'Classifier {clf_type} score: {grid_clf.best_score_}')
    return estimated_clf


def plot_multiple_roc_curves(clfs, X_train, y_train, X_test, y_test, ticker, model_type=3):
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1])
    for clf in clfs:
        best_clf = clfs[clf].fit(X_train, y_train)
        y_pred = best_clf.predict_proba(X_test)[:, 1]
        fpr_test, tpr_test, thresholds_test = roc_curve(y_test, best_clf.predict_proba(X_test)[:, 1])
        print(clf, roc_auc_score(y_test, y_pred))
        plt.plot(fpr_test, tpr_test)
    plt.legend(['equality'] + list(clfs.keys()))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {ticker}')
    plt.savefig(f'data/results/grid_cv/{ticker}_model{model_type}_all_roc.png')
    plt.close()


def plot_roc_curves(best_clf, X_train, y_train, X_test, y_test, ticker, model_type):
    clf_type = best_clf.__class__.__name__
    if clf_type == 'LinearSVC':
        return
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, best_clf.predict_proba(X_test)[:, 1])
    fpr_tr, tpr_tr, thresholds_tr = roc_curve(y_train, best_clf.predict_proba(X_train)[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_test, tpr_test)
    plt.plot(fpr_tr, tpr_tr)
    plt.plot([0, 1], [0, 1])
    plt.legend(['test', 'train', 'equality'])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve {clf_type} - Score: {round(roc_auc_score(y_test, best_clf.predict(X_test)), 2)}')
    plt.savefig(f'data/results/grid_cv/{ticker}_{clf_type}_model{model_type}_roc.png')
    plt.close()


def get_data_by_model(ticker, clf_types, X_train, X_test, model_type):
    if model_type == 1:
        return {clf_type: X_train for clf_type in clf_types}, {clf_type: X_test for clf_type in clf_types}

    training_data = {}
    test_data = {}

    if model_type == 2:
        base_path = f'data/results/forward_selection/{ticker}/'
        fwd_features_path = f'{base_path}features_25%_R39.csv'
        fwd_features = pd.read_csv(fwd_features_path, index_col=0)
        fwd_results_path = f'{base_path}results_25%_R39.csv'
        fwd_results = pd.read_csv(fwd_results_path, index_col=0)
        for clf_type in clf_types:
            result_col = [c for c in fwd_results.columns if clf_type in c][0]
            idx = fwd_results[result_col].argmax()
            feature_col = [c for c in fwd_features.columns if clf_type in c][0]
            features = fwd_features[feature_col].iloc[:idx + 1].to_list()
            training_data[clf_type] = X_train[features]
            test_data[clf_type] = X_test[features]

    elif model_type == 3:
        data_path = f'data/results/exhaustive_selection/{ticker}_25%_R39.csv'
        data = pd.read_csv(data_path, index_col=0)
        for clf_type in clf_types:
            features = [f'{ticker}_returns_sr', f'{ticker}_momentum_im', f'{ticker}_momentum_hlm']
            clf_col = [c for c in data.columns if clf_type in c][0]
            idx = data[clf_col].argmax()
            rtn, mtm, sma, std = data.index[idx].split('_')[1::2][:-1]
            features += [f'{ticker}_returns_{rtn}', f'{ticker}_momentum_{mtm}',
                         f'{ticker}_sma_{sma}', f'{ticker}_std_{std}']
            training_data[clf_type] = X_train[features]
            test_data[clf_type] = X_test[features]

    return training_data, test_data


def evaluate_classifier(best_clf, X_train, y_train, X_test, y_test, ticker, model_type=1):
    best_params = best_clf.get_params()
    try:
        coefs = best_clf._coef
    except AttributeError:
        coefs = None
    plot_roc_curves(best_clf, X_train, y_train, X_test, y_test, ticker, model_type=model_type)
    y_pred = best_clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    clf_type = best_clf.__class__.__name__
    results = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
               'acc': accuracy_score(y_test, y_pred), 'precision': precision_score(y_test, y_pred),
               'recall': recall_score(y_test, y_pred), 'f1': f1_score(y_test, y_pred)}
    results = pd.DataFrame(results.values(), index=results.keys(), columns=[f'{ticker}_{clf_type}'])
    return best_params, coefs, results


def run_all_grid_search(ticker, grid_params, clf_types, X_train, y_train, X_test, y_test, model_types=None):
    if model_types is None:
        model_types = [1, 2, 3]
    best_params = {}
    coefs = {}
    results = pd.DataFrame()

    for model_type in model_types:
        print(model_type)
        training_data, test_data = get_data_by_model(ticker, clf_types, X_train, X_test, model_type)
        for clf_type in clf_types:
            print(clf_type)
            id_ = f'{ticker}_{model_type}_{clf_type}'
            X_train_ = training_data.get(clf_type)
            X_test_ = test_data.get(clf_type)
            best_clf = grid_search(clf_type=clf_type, grid_params=grid_params, X_train=X_train_, y_train=y_train)
            prms, cfs, rslts = evaluate_classifier(best_clf=best_clf, X_train=X_train_, y_train=y_train, X_test=X_test_,
                                                   y_test=y_test, ticker=ticker, model_type=model_type)
            best_params[id_] = prms
            coefs[id_] = cfs
            results = pd.concat([results, rslts], axis=1)
    results.to_csv(f'data/results/grid_cv/{ticker}_stratified_results.csv')
    pickle.dump(coefs, open(f'data/results/grid_cv/{ticker}_stratified_coefs.pkl', 'wb'))
    pickle.dump(best_params, open(f'data/results/grid_cv/{ticker}_stratified_params.pkl', 'wb'))


def plot_up_down_scatter(up, down, ticker, days, name=''):
    plt.figure(figsize=(6, 4))
    plt.scatter(up['returns_1d'], up[f'momentum_{days}d'], color='skyblue', s=3)
    plt.scatter(down['returns_1d'], down[f'momentum_{days}d'], color='maroon', s=3)
    plt.title(f'{days}d Momentum vs 1d Return - {ticker} {name}')
    plt.xlabel('1d Return')
    plt.ylabel(f'{days}d Momentum')
    plt.legend(['Up', 'Down'])
    plt.savefig(f'data/results/svm/{days}d_mom_1d_rtn_{ticker}_{name}.png')
    plt.close()


def get_backtesting_data(data, training_start, test_start, output_column='result_zero_one', renormalise=True):
    data.index = [datetime.datetime.strptime(d, '%d/%m/%Y') for d in data.index]
    train_cols = data.columns.to_list()
    test_column = [c for c in train_cols if output_column in c]
    if len(test_column) != 1:
        raise Exception('Should be precisely one result column.')
    test_column = test_column[0]
    train_cols.remove(test_column)
    X, y = data[train_cols], data[test_column]
    if renormalise:
        X = renormalise_data(X)
    X_train = X.loc[(X.index >= training_start) & (X.index < test_start)]
    X_test = X.loc[X.index >= test_start]
    y_train = y.loc[(y.index >= training_start) & (y.index < test_start)]
    y_test = y.loc[y.index >= test_start]
    return X_train, y_train, X_test, y_test


def compute_strategy_nav(y_test, y_pred, probability, scale_factor=0.1):
    pnl = (2 * (y_test == y_pred) - 1) * abs(2 * probability - 1) * scale_factor
    return (1 + pnl).cumprod()


def plot_backtest_nav(nav, clf_type, ticker):
    plt.figure(figsize=(8, 6))
    plt.plot(nav.index, nav)
    plt.xlabel('Date')
    plt.ylabel('NAV')
    live_start = datetime.datetime.strftime(nav.index[0], format='%Y-%m-%d')
    live_end = datetime.datetime.strftime(nav.index[-1], format='%Y-%m-%d')
    plt.title(f'{ticker} {clf_type} {live_start} - {live_end} NAV')
    plt.savefig(f'data/results/bt/{ticker}_{clf_type}_{live_start}_{live_end}.png')
    plt.close()


RANDOM_STATE = 39
test_size = 0.25
tickers = ('pl1',)
clf_types = ('nb', 'knn', 'svm', 'logit')

clf_params = {'logit': {'penalty': 'none', 'solver': 'saga', 'max_iter': 100},
              'svm': {'C': 1},
              'nb': {'alpha': 1},
              'knn': {'n_neighbors': 5}}

# forward and exhaustive selection

run_forward_selection(tickers=tickers, test_size=test_size, RANDOM_STATE=RANDOM_STATE)
run_exhaustive_selection(tickers, test_size=test_size, RANDOM_STATE=RANDOM_STATE)
create_forward_selection_analysis(tickers)

# Grid Search

C_search = [100, 10, 1, 0.1, 0.01]
grid_params = {'logit': {'penalty': ['none', 'l1', 'l2'], 'C': C_search, 'max_iter': [1000]},
               'svm': {'C': C_search, 'dual': [False], 'penalty': ['l1', 'l2']},
               'svm_': {'C': [0.01]},
               'knn': {'n_neighbors': range(1, 31, 2), 'metric': ['manhattan', 'euclidean']},
               'nb': {'alpha': [0.1 * x for x in range(1, 11)]}}

for ticker in tickers:
    test_size = 0.25
    data = get_feature_data(ticker, drop_features=('ewma',), after_year=None)
    X_train, X_test, y_train, y_test = get_split_data(data=data, test_size=test_size, output_column='result_zero_one')
    run_all_grid_search(ticker, grid_params, clf_types, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

# Logistic regression regularisation demonstration

test_size = 0.25
ticker = 'hg1'
model_type = 3
if not os.path.exists('data/results/logistic_regression/'):
    os.mkdir('data/results/logistic_regression/')

data = get_feature_data(ticker, drop_features=('ewma',), after_year=None)
X_train, X_test, y_train, y_test = get_split_data(data=data, test_size=test_size, output_column='result_zero_one')
coefs = pd.DataFrame(index=X_train.columns)
accuracy = {}
for penalty, C in [('none', 1), ('l1', 10), ('l1', 1), ('l1', 0.1), ('l2', 10), ('l2', 1), ('l2', 0.1)]:
    id_ = f'penalty_{penalty}_C_{C}'
    clf = get_classifier('logit', params={'penalty': penalty, 'C': C, 'solver': 'saga'})
    clf.fit(X_train, y_train)
    coefs[id_] = clf.coef_[0]
    accuracy[id_] = accuracy_score(y_test, clf.predict(X_test))
coefs.to_csv('data/results/logistic_regression/reg_coefs.csv')
pd.Series(accuracy).to_csv('data/results/logistic_regression/accuracy.csv')

# SVM

if not os.path.exists('data/results/svm/'):
    os.mkdir('data/results/svm/')
for ticker in ('hg1',):
    data = get_feature_data(ticker, drop_features=('ewma',), after_year=None)
    X_train, X_test, y_train, y_test = get_split_data(data=data, test_size=test_size, output_column='result_zero_one',
                                                      renormalise=False)

    X_svm = X_train[[f'{ticker}_momentum_{k}d' for k in range(1, 6)] + [f'{ticker}_returns_1d']]
    X_svm = pd.concat([X_svm, y_train], axis=1)
    X_svm.columns = [f'momentum_{k}d' for k in range(1, 6)] + ['returns_1d', 'up_down']
    up_svm = X_svm.loc[X_svm.up_down == 1].drop('up_down', axis=1)
    down_svm = X_svm.loc[X_svm.up_down == 0].drop('up_down', axis=1)

    for d in range(1, 6):
        plot_up_down_scatter(up=up_svm, down=down_svm, ticker=ticker, days=d)

    cols = [f'{ticker}_returns_1d', f'{ticker}_momentum_1d', f'{ticker}_result_zero_one',
            f'{ticker}_returns_sr', f'{ticker}_sma_5d', f'{ticker}_std_20d']
    X_train, X_test, y_train, y_test = get_split_data(data=data[cols], test_size=test_size,
                                                      output_column='result_zero_one', renormalise=True)
    for c in [100, 10, 1, 0.1, 0.01]:
        clf = SVC(kernel='linear', C=c)
        clf.fit(X_train, y_train)
        y_pred = pd.DataFrame(clf.predict(X_test), index=X_test.index)
        X_sv = pd.concat([X_test[[f'{ticker}_returns_1d', f'{ticker}_momentum_1d']], y_pred], axis=1)
        X_sv.columns = ['returns_1d', 'momentum_1d', 'up_down']
        up_sv = X_sv.loc[X_sv.up_down == 1].drop('up_down', axis=1)
        down_sv = X_sv.loc[X_sv.up_down == 0].drop('up_down', axis=1)
        plot_up_down_scatter(up=up_sv, down=down_sv, ticker=f'{ticker}', days=1, name=f'C_{c}')

# kNN

# code closely adapted from:
# https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html

if not os.path.exists('data/results/knn/'):
    os.mkdir('data/results/knn/')

for ticker in ('pl1',):
    for neighbours in [5, 10, 20, 100]:
        data = get_feature_data(ticker, drop_features=('ewma',), after_year=None)
        X_train, X_test, y_train, y_test = get_split_data(data=data, test_size=test_size,
                                                          output_column='result_zero_one',
                                                          renormalise=True)
        h = .02
        clf = KNeighborsClassifier(n_neighbors=neighbours)
        X_knn = X_train[[f'{ticker}_std_20d', f'{ticker}_returns_5d', ]]
        clf.fit(X_knn, y_train)
        x_min, x_max = X_knn.iloc[:, 0].min() - 1, X_knn.iloc[:, 0].max() + 1
        y_min, y_max = X_knn.iloc[:, 1].min() - 1, X_knn.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        cmap_light = ListedColormap(['moccasin', 'cornflowerblue'])
        cmap_bold = ListedColormap(['darkorange', 'darkblue'])

        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X_knn.iloc[:, 0], X_knn.iloc[:, 1], c=y_train, s=20, cmap=cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        handles = [mpatches.Patch(color='cornflowerblue', label='Up'), mpatches.Patch(color='moccasin', label='Down')]
        handles += [mpatches.Patch(color='darkblue', label='Up Train'),
                    mpatches.Patch(color='darkorange', label='Down Train')]
        plt.xlabel(X_knn.columns[0])
        plt.ylabel(X_knn.columns[1])
        plt.legend(handles=handles, loc=2)
        plt.title(f'kNN Decision Boundary - {ticker} k = {neighbours}')
        plt.savefig(f'data/results/knn/{ticker}_{neighbours}_neighbours.png')

# plot all roc curves

for ticker in tickers:
    data = get_feature_data(ticker, drop_features=('ewma',), after_year=None)
    X_train, X_test, y_train, y_test = get_split_data(data=data, test_size=test_size, output_column='result_zero_one',
                                                      renormalise=True)

    clfs = {'nb': BernoulliNB(alpha=0.1),
            'knn': KNeighborsClassifier(n_neighbors=5, metric='manhattan'),
            'svm_': SVC(probability=True, kernel='linear', random_state=RANDOM_STATE),
            'logit': lm.LogisticRegression(random_state=RANDOM_STATE, C=0.1, penalty='l1')}

    plot_multiple_roc_curves(clfs, X_train, y_train, X_test, y_test, ticker)

# strategy backtesting

for ticker in tickers:
    clfs = {'nb': BernoulliNB(alpha=0.1),
            'logit': lm.LogisticRegression(random_state=RANDOM_STATE, C=0.1, penalty='l1')}
    clf_types = ['nb', 'logit']

    data = get_feature_data(ticker, drop_features=('ewma',), after_year=2016)
    train_start = datetime.datetime(2016, 4, 30)
    test_start = datetime.datetime(2019, 4, 30)
    X_train, y_train, X_test, y_test = get_backtesting_data(data, training_start=train_start, test_start=test_start)
    training_data, test_data = get_data_by_model(ticker, clf_types, X_train, X_test, model_type=3)
    trained_cls = {}
    for clf_type in clfs:
        c = clfs[clf_type]
        c.fit(X_train, y_train)
        p = c.predict_proba(X_test)[:, 1]
        y_pred = c.predict(X_test)
        nav = compute_strategy_nav(y_test, y_pred, p, scale_factor=0.10)
        plot_backtest_nav(nav, clf_type, ticker)
