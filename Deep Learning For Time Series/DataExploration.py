import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from numpy.linalg import eig
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from collections import OrderedDict

plt.style.use('ggplot')


class DataExploration(object):
    def __init__(self, ticker, num_estimators=64):
        self.ticker = ticker
        feature_path = f'data/feature_data/{ticker}_normalised_features.csv'
        self.features = pd.read_csv(feature_path, index_col=0)
        self.training_data = self.features.iloc[:, :-1]
        self.test_data = self.features.iloc[:, -1]
        self.set_feature_types()
        self.num_estimators = num_estimators

        self.result_path = 'data/exploration/'
        self.corr_path = self.result_path + 'corr/'
        self.pca_path = self.result_path + 'pca/'
        self.dt_path = self.result_path + 'dt/'

    def set_feature_types(self):
        """
        Set list of feature families e.g. ['returns', 'volumes', ...]
        Useful to have quick access to this

        Returns:
            None
        """
        cols = self.training_data.columns.to_list()
        fts = ['_'.join(f.split('_')[1:]) for f in cols]
        core_fts = ['_'.join(f.split('_')[:-1]) for f in fts]
        self.feature_types = sorted(list(set([cf for cf in core_fts])))

    @staticmethod
    def get_feature_type(ft):
        """

        Args:
            ft (str): feature name as taken from column of features dataframe

        Returns:
            (str): feature family name - an element of self.feature_types
        """
        return '_'.join(ft.split('_')[1:-1])

    def get_top_features(self, feature_importance, training_data, top_n, max_count):
        """

        Args:
            feature_importance (list): importance scores corresponding to features in training data
            training_data (pd.DataFrame): dataframe consisting of the features in consideration
            top_n (int): number of most important features to select
            max_count (int): maximum number of features that can be selected from any one feature family

        Returns:
            (pd.DataFrame): dataframe indexed by feature, values being importance numbers
        """
        d = {feature_importance[idx]: training_data.columns[idx] for idx in range(len(feature_importance))}
        d = OrderedDict(sorted(d.items(), reverse=True))
        top_features = []
        feature_type_count = {ft: 0 for ft in self.feature_types}
        top_count = 0

        for k, v in d.items():
            ft_type = self.get_feature_type(v)
            if feature_type_count[ft_type] < max_count:
                top_features.append(v)
                top_count += 1
                if top_count >= top_n:
                    break
            feature_type_count[ft_type] += 1

        d_reverse = {v: k for k, v in d.items()}
        result = pd.DataFrame([d_reverse[k] for k in top_features], index=top_features,
                              columns=[f'{self.ticker}_importance'])
        return result

    def rf_feature_selection(self, top_n, max_count=3, file_name=None, training_data=None, ):
        """
        Fit random forest to estimate most important features

        Args:
            top_n (int): number of features to keep, ordered from most to least important

        Returns:
            (pd.DataFrame): dataframe indexed by feature name, values being feature importance
        """
        # fit random
        if file_name is None:
            file_name = f'{self.dt_path}ft_importance/{self.ticker}_top_{top_n}_max_{max_count}_rf_{self.num_estimators}.csv'

        if not os.path.exists(file_name):
            selection = SelectFromModel(RandomForestClassifier(n_estimators=self.num_estimators))
            if training_data is None:
                training_data = self.training_data
            else:
                max_count = top_n
            selection.fit(training_data, self.test_data)
            feature_importance = selection.estimator_.feature_importances_

            top_features = self.get_top_features(feature_importance, training_data, top_n=top_n, max_count=max_count)
            top_features.to_csv(file_name)

    # Variants of compute_correlations, perform_pca and compute_pca
    # were also used in CQF assignment 3
    def compute_correlations(self):
        self.features.corr().to_csv(f'{self.corr_path}{self.ticker}_overall_corr.csv')
        avg_corr = {}
        for ft in self.feature_types:
            # restrict to relevant features
            columns = [c for c in self.training_data.columns if '_'.join(c.split('_')[1:-1]) == ft]
            if len(columns) != 1:
                data = self.features[columns]
                data.columns = [c.split('_')[-1] for c in data.columns]
                corr = data.corr()
                corr.to_csv(f'{self.corr_path}{self.ticker}_{ft}.csv')

                # plot heatmap
                plt.figure()
                sns.heatmap(corr)
                plt.title(f'{self.ticker} {ft}')
                plt.savefig(f'{self.corr_path}{self.ticker}_{ft}.png')
                plt.close()

                # compute feature average correlations
                n = corr.shape[0]
                avg_corr[ft] = (corr.sum().sum() - n) / (n * (n - 1))
        avg_corr = pd.DataFrame(avg_corr.values(), avg_corr.keys(), columns=['avg_correl'])
        avg_corr.sort_values(by='avg_correl').to_csv(f'{self.corr_path}{self.ticker}_average_feature_correl.csv')

    def perform_pca(self):
        corr = self.features.corr()
        values, vectors = eig(corr)
        return values / sum(values)

    def compute_pca(self):
        pca = self.perform_pca()
        pca_cumulative = 100 * pca.cumsum()
        s = pd.Series(pca_cumulative)
        s.index = range(1, len(s) + 1)
        s.to_csv(f'{self.pca_path}{self.ticker}_pca_cumulative.csv')

        plt.figure(figsize=(12, 8))
        plt.plot(range(1, len(pca) + 1), pca_cumulative, 'x')
        plt.legend([self.ticker])
        plt.xlabel('Eigenvector Number')
        plt.ylabel('Cumulative % Variance')
        plt.savefig(f'{self.pca_path}{self.ticker}_contributions_to_variance.png')
        plt.close()


class DecisionTreeHelpers(object):
    """
    Uses Decision Tree Classifiers and Random Forests to determine feature importance
    """

    def __init__(self, tickers, num_ests, max_counts, top_ns, rf_result_path='data/exploration/rf_cv_scores.csv'):
        """

        Args:
            tickers (list): Tickers of interest
            num_ests (list): Number of trees to be grown in random forest
            max_counts (list): Maximum number of features of each type that can be included in important features
            top_ns (list): Number of top features to be selected
            rf_result_path (str): Location to save result of final analysis containing accuracy scores
        """
        self.tickers = tickers
        self.num_ests = num_ests
        self.max_counts = max_counts
        self.top_ns = top_ns
        self.rf_result_path = rf_result_path

    def create_rf_feature_importance_files(self):
        """
        Save feature importance file for each possible combination of
        num_estimators, top_n and max_count

        Returns:
            None
        """
        for t in self.tickers:
            for num_est in self.num_ests:
                data_explorer = DataExploration(ticker=t, num_estimators=num_est)
                for max_count in self.max_counts:
                    for top_n in self.top_ns:
                        print(t, num_est, max_count, top_n)
                        data_explorer.rf_feature_selection(top_n=top_n, max_count=max_count)

    @staticmethod
    def get_filtered_feature_importance(ticker, top_n, max_count, num_est):
        ft_path = f'data/exploration/dt/ft_importance/{ticker}_top_{top_n}_max_{max_count}_rf_{num_est}.csv'
        fts = pd.read_csv(ft_path, index_col=0).index.to_list()
        data_explorer = DataExploration(ticker=ticker, num_estimators=num_est)
        training_data = data_explorer.training_data[fts]
        file_name = f'data/exploration/dt/{ticker}_final_ft_importance_top_{top_n}_max_{max_count}_rf_{num_est}.csv'
        return data_explorer.rf_feature_selection(top_n=top_n, max_count=max_count, file_name=file_name,
                                                  training_data=training_data)

    def add_rf_results(self, ticker, rf_results, X, y, cv, clf_rf):
        """

        Args:
            ticker (str):
            rf_results:
            X (pd.DataFrame): training features
            y (pd.Series): class labels
            cv (sklearn.cross_val_score): cross vlaidation object
            clf_rf (sklearn.RanfomForestClassifier): classifier object

        Returns:

        """
        for num_est in self.num_ests:
            for max_count in self.max_counts:
                for top_n in self.top_ns:
                    print(num_est, max_count, top_n)

                    ft_path = f'data/exploration/dt/ft_importance/{ticker}_top_{top_n}_max_{max_count}_rf_{num_est}.csv'
                    fts = pd.read_csv(ft_path, index_col=0).index.to_list()
                    rf_cv_score = cross_val_score(clf_rf, X[fts], y, cv=cv)
                    rf_results[f'top_{top_n}_max_{max_count}_rf_{num_est}'] = [rf_cv_score.mean(), rf_cv_score.std()]

        rf_result = pd.DataFrame(rf_results).transpose()
        rf_result.columns = [f'{ticker}_accuracy', f'{ticker}_std']
        return rf_result

    def train_rf_feature_importance(self):
        result = pd.DataFrame()

        for t in self.tickers:
            rf_results = {}
            data_explorer = DataExploration(ticker=t)
            X, y = data_explorer.training_data, data_explorer.test_data

            clf = DecisionTreeClassifier(random_state=0)
            clf_rf = RandomForestClassifier(max_depth=5, random_state=0)
            cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

            cv_score = cross_val_score(clf, X, y, cv=cv)
            rf_cv_score = cross_val_score(clf_rf, X, y, cv=cv)
            rf_results['dt'] = [cv_score.mean(), cv_score.std()]
            rf_results['full'] = [rf_cv_score.mean(), rf_cv_score.std()]

            rf_result = self.add_rf_results(ticker=t, rf_results=rf_results, X=X, y=y, cv=cv, clf_rf=clf_rf)
            result = pd.concat([result, rf_result], axis=1)

        return result

    def save_rf_result(self):
        """
        Save pd.DataFrame containing accuracy of random forest trained on each of the
        combinations of most important features determined by the class' parameters

        Returns:
            None
        """
        result = self.train_rf_feature_importance()
        for col_name, split_index in [('top', 1), ('max', 3), ('rf', 5)]:
            result[col_name] = [x.split('_')[split_index] if '_' in x else 0 for x in result.index]
        result.to_csv(self.rf_result_path)
