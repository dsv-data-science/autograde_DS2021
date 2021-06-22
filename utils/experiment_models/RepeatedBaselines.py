from utils.experiment_models.RepeatedHoldOut import RepeatedHoldOut
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
import numpy as np


class RepeatedBaselines(RepeatedHoldOut):
    """
    RepeatedBaselines is a wrapper class that allows for easy implementations of repeated hold-out experiment_models using the
    sklearn library models; such as LogisticRegression and RandomForestClassifier.

    Attributes
    ----------
    models : dict
        The models to use for repeated hold-out, given as a dict of key-value pairs where the key is the model name and
        the value is the model object. E.g: <'Logistic Regressor', LogisticRegression>
    metrics : list
        What metrics to include in the results
    iterations : int, optional
        The amount of iterations of repeated hold-out (Default is 10)
    random_state: int, optional
        Used for reproducibility (Default is None)

    Methods
    -------
    repeated_split(self, X=None, y=None, test_size=0.15, valid_size=None, stratify=None)
        Splits the samples into iterations number of splits and returns a list of the splits.
    fit_predict(self, split_list=None, correlation_data=None)
        Fit and predict data for all models and for each split in split_list. Metrics are computed for each split.
    convert_data(split_list=None, representation=None)
        Converts the data in split_list into a representation and returns a converted split_list.
        This could, for example, be a count-vectorizer or a tf-idf-vectorizer.
    resample(split_list=None, resampling_techs=None)
        Resamples each split in a split_list using a resampling technique given by the parameter resampling_techs.
        Returns a resampled split_list.
    """

    AVERAGE_PRECISION = 'average_precision'
    ROC_AUC = 'roc_auc'
    SPEARMAN = 'spearman'
    CLASS_REPORT = 'averaged_classification_report'

    SUPPORTED_METRICS = [AVERAGE_PRECISION,
                         ROC_AUC,
                         SPEARMAN,
                         CLASS_REPORT]

    def __init__(self,
                 models=None,
                 metrics=None,
                 iterations=10,
                 random_state=None):
        """
        Parameters
        ----------
        models : dict
            The models to use for repeated hold-out, given as a dict of key-value pairs where the key is the model name and
            the value is the model object. E.g: <'Logistic Regressor', LogisticRegression>
        metrics : list
            What metrics to include in the results
        iterations : int, optional
            The amount of iterations of repeated hold-out (Default is 10)
        random_state: int, optional
            Used for reproducibility (Default is None)

        Raises
        ------
        ValueError
            If parameter models is None or empty
        ValueError
            If metrics is None, empty, or not included in supported metrics.
        """
        super().__init__(metrics, iterations, random_state)
        if models is None or not models:
            raise ValueError('At least one model needs to be passed to the models parameter.')
        self.models = models

    def repeated_split(self, X=None, y=None, test_size=0.15, valid_size=None, stratify=None):
        """Takes samples and returns repeated splits of the samples in the form of a list.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        test_size : float, optional
            Represents the proportion of the dataset to include in the test split (Default is 0.15).
        valid_size : float, optional
            Represents the proportion of the dataset to include in the validation split (Default is None). If valid_size
            is None, the samples will be split into a training and test set per split. If valid_size is not None, then
            samples will be split into a training, validation, and test set per split.
        stratify: array-like, optional
            If not None, data is split in a stratified fashion, using the array-like stratify parameter
            as class labels (Default is None).

        Returns
        ----------
        list
            A list of the splits as such: [split1, split2, split3, split4..].
        """

        if valid_size is None:  # if valid_size is None, then return a list of repeated splits without validation sets
            return super().repeated_split(X=X, y=y, test_size=test_size, stratify=stratify)

        train_ratio = 1 - (test_size + valid_size)
        valididation_ratio = test_size / (test_size + valid_size)
        split_list = super().repeated_split(X=X, y=y, test_size=1 - train_ratio, stratify=stratify)

        random_state = self.random_state
        train_dev_test_list = []
        for i in range(0, len(split_list)):
            x_train, x_test, y_train, y_test, _, idx1 = split_list[i]

            indices = np.arange(np.array(x_test).shape[0])

            x_dev, x_test, y_dev, y_test, _, idx2 = train_test_split(x_test,
                                                                     y_test,
                                                                     indices,
                                                                     test_size=valididation_ratio,
                                                                     stratify=(
                                                                         y_test if stratify is not None else None),
                                                                     random_state=(
                                                                         random_state + i if random_state else None))

            train_dev_test_list.append([x_train, y_train,
                                        x_dev, y_dev,
                                        x_test, y_test,
                                        idx1, idx2])

        return train_dev_test_list

    def fit_predict(self, split_list=None, correlation_data=None):
        """Takes a list of repeated splits given by repeated_splits() and fits as well as predicts the splits
        for all models. Metrics are computed for each split.

        Parameters
        ----------
        split_list : list
            A list of repeated splits in the form given by repeated_splits()
        correlation_data : array-like of shape (n_samples,)
            If metrics contain spearman correlation then the correlation between correlation_data and
            the predicted probabilities are computed. This parameter is mandatory if spearman correlation has been
            given as a metric to be computed.

        Returns
        ----------
        dict
            a dictionary of computed metrics for all models.
        """
        super().fit_predict(split_list, correlation_data)
        i = 1
        for x_train, x_test, y_train, y_test, idx1, idx2 in split_list:
            self._train(x_train, y_train)
            self._predict(x_test, y_test, idx1, idx2)
            i = i + 1

        if self.CLASS_REPORT in self.metrics:
            for model_name in self.models.keys():
                model_scores = self.model_predictions.get(model_name, None)
                all_reports = model_scores[self.CLASS_REPORT]
                averaged_report = RepeatedHoldOut._avg_classification_reports(all_reports)
                model_scores[self.CLASS_REPORT] = averaged_report

        return self.model_predictions

    def _train(self, x_train=None, y_train=None):
        """Fit x_train and y_train on each model
        """
        models = self.models
        for _, model in models.items():
            model.fit(x_train, y_train)

    def _predict(self, x_test=None, y_test=None, idx1=None, idx2=None):
        """Predict x_test on each model and calculate metric scores
        """
        models = self.models
        for model_name, model in models.items():
            predictions = model.predict(x_test)
            predict_probas = model.predict_proba(x_test)
            self._scores(model_name, y_test, predictions, predict_probas, idx1, idx2)

    def _scores(self, model_name, y_true=None, predictions=None, predict_probas=None, idx1=None, idx2=None):
        """Calculates all metric scores
        """
        model_scores = self.model_predictions.get(model_name, None)
        corr_data = self.correlation_data
        super()._calc_n_add_scores(model_scores, y_true, predictions, predict_probas, idx1, idx2, corr_data)

    def _reset_scores(self):
        """Resets the model_predictions instance variable. This variable includes the metric scores of each split
        for each model. This method is called on each call to fit_predict().
        """
        model_predictions = {}
        for model_name in self.models.keys():
            for metric in self.metrics:
                model_predictions.setdefault(model_name, {})[metric] = []
        self.model_predictions = model_predictions

    def _spearman(y_true=None, _predictions=None, predict_probas=None, idx1=None, idx2=None, correlation_data=None):
        """Calculates the spearman correlation between the predicted probabilities and the correlation data.
        """
        first_split = correlation_data.iloc[idx1]
        second_split = first_split.iloc[idx2]
        corr, p_val = spearmanr(predict_probas[:, 1], second_split)
        return [corr, p_val]

    @staticmethod
    def convert_data(split_list=None, representation=None):
        """Converts the data in split_list into a representation and returns a converted split_list.
        Only supports Sklearn representation classes such as CountVectorizer or TFIdfVectorizer.

        Parameters
        ----------
        split_list : list
            A list of repeated splits in the form given by repeated_splits()
        representation : Object
            Only supports Sklearn representation classes such as CountVectorizer or TFIdfVectorizer.

        Returns
        ----------
        list
            a converted split_list
        """
        converted_list = []
        for x_train, x_test, y_train, y_test, idx1, idx2 in split_list:
            representation.fit(x_train)
            x_train = representation.transform(x_train)
            x_test = representation.transform(x_test)
            converted_list.append([x_train, x_test, y_train, y_test, idx1, idx2])
        return converted_list

    METRICS_FUNCTIONS = {AVERAGE_PRECISION: RepeatedHoldOut._average_prec,
                         ROC_AUC: RepeatedHoldOut._roc_auc,
                         SPEARMAN: _spearman,
                         CLASS_REPORT: RepeatedHoldOut._classification_report}