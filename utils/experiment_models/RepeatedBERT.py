from utils.experiment_models.RepeatedHoldOut import RepeatedHoldOut
from scipy.stats import spearmanr
import ktrain
from ktrain import text
from sklearn.model_selection import train_test_split
import numpy as np


class RepeatedBERT(RepeatedHoldOut):
    """
    RepeatedBERT is a wrapper class that allows for easy implementations of repeated hold-out experiment_models using the
    ktrain library.

    Attributes
    ----------
    transformer : Transformer
        The transformer to use.
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
        Fit and predict data on each split in split_list. Metrics are computed for each split.
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
                 transformer=None,
                 metrics=None,
                 iterations=10,
                 random_state=None):
        """
        Parameters
        ----------
        transformer : Transformer
            The transformer to use.
        metrics : list
            What metrics to include in the results
        iterations : int, optional
            The amount of iterations of repeated hold-out (Default is 10)
        random_state: int, optional
            Used for reproducibility (Default is None)

        Raises
        ------
        ValueError
            If metrics is None, empty, or not included in supported metrics.
        """
        super().__init__(metrics, iterations, random_state)
        self.transformer = transformer

    def repeated_split(self, X=None, y=None, test_size=0.15, valid_size=0.15, stratify=None):
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

            train_dev_test_list.append([list(x_train), list(y_train),
                                        list(x_dev), list(y_dev),
                                        list(x_test), list(y_test),
                                        idx1, idx2])

        return train_dev_test_list

    def fit_predict(self, split_list=None, batch_size=6, lr=2e-5, epochs=30, correlation_data=None):
        """Takes a list of repeated splits given by repeated_splits() and fits as well as predicts the splits.
        Metrics are computed for each split.

        Parameters
        ----------
        split_list : list
            A list of repeated splits in the form given by repeated_splits()
        batch_size : int, optional
            The batch_size to use (Default is 6)
        lr : float, optional
            The learning rate to use (Default is 2e-5)
        epochs : int, optional
            The number of epochs to use (Default is 30)
        correlation_data : array-like of shape (n_samples,)
            If metrics contain spearman correlation then the correlation between correlation_data and
            the predicted probabilities are computed. This parameter is mandatory if spearman correlation has been
            given as a metric to be computed.
        Returns
        ----------
        dict
            a dictionary of computed metrics on all splits.
        """
        super().fit_predict(split_list, correlation_data)
        for x_train, y_train, x_dev, y_dev, x_test, y_test, idx1, idx2 in split_list:
            self._train(x_train, y_train, x_dev, y_dev, x_test, y_test, idx1, idx2, batch_size, lr, epochs)

        if self.CLASS_REPORT in self.metrics:
            metric_map = self.model_predictions
            all_reports = metric_map[self.CLASS_REPORT]
            averaged_report = RepeatedHoldOut._avg_classification_reports(all_reports)
            metric_map[self.CLASS_REPORT] = averaged_report

        return self.model_predictions

    def _train(self, x_train, y_train, x_dev, y_dev, x_test, y_test, idx1, idx2, batch_size, lr, epochs):
        """Train XLM-R"""
        transformer = self.transformer
        trn = transformer.preprocess_train(x_train, y_train)
        val = transformer.preprocess_test(x_dev, y_dev)
        model = transformer.get_classifier()
        learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=batch_size)
        learner.fit_onecycle(lr, epochs)
        learner.validate(class_names=transformer.get_classes())
        predictor = ktrain.get_predictor(learner.model, preproc=transformer)
        self._predict(predictor, x_test, y_test, idx1, idx2)
        del trn, val, model, learner, predictor  # delete all variables not needed anymore to save space

    def _predict(self, predictor=None, x_test=None, y_test=None, idx1=None, idx2=None):
        """Predict x_test using predictor and calculate metric scores"""
        predictions = predictor.predict(x_test)
        predict_probas = predictor.predict_proba(x_test)
        self._scores(y_test, predictions, predict_probas, idx1, idx2)

    def _scores(self, y_true=None, predictions=None, predict_probas=None, idx1=None, idx2=None):
        """Calculate metric scores"""
        metric_map = self.model_predictions
        corr_data = self.correlation_data
        super()._calc_n_add_scores(metric_map, y_true, predictions, predict_probas, idx1, idx2, corr_data)

    def _reset_scores(self):
        """Resets the model_predictions instance variable. This variable includes the metric scores of each split.
        The method is called on each call to fit_predict().
        """
        model_predictions = {}
        for metric in self.metrics:
            model_predictions.setdefault(metric, [])

        self.model_predictions = model_predictions

    def _spearman(y_true=None, _predictions=None, predict_probas=None, idx1=None, idx2=None, correlation_data=None):
        """Calculates the spearman correlation between the predicted probabilities and the correlation data.
        """
        first_split = [correlation_data[i] for i in idx1]
        second_split = [first_split[i] for i in idx2]
        corr, p_val = spearmanr(predict_probas[:, 1], second_split)
        return [corr, p_val]

    METRICS_FUNCTIONS = {AVERAGE_PRECISION: RepeatedHoldOut._average_prec,
                         ROC_AUC: RepeatedHoldOut._roc_auc,
                         SPEARMAN: _spearman,
                         CLASS_REPORT: RepeatedHoldOut._classification_report}