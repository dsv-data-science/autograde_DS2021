from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from statistics import mean


class RepeatedHoldOut:
    """RepeatedHoldOut is an abstract class used for repeated hold-out experiment_models.

    Child classes include:
    - RepeatedBaselines
    - RepeatedBERT
    """

    def __init__(self,
                 metrics=None,
                 iterations=10,
                 random_state=None):
        if type(self) is RepeatedHoldOut:
            raise Exception("RepeatedHoldOut is an abstract class and cannot be instantiated directly.")

        self.metrics = metrics if self._is_valid_metrics(metrics) else None
        self.iterations = iterations
        self.random_state = random_state

    def repeated_split(self, X=None, y=None, test_size=0.15, stratify=None):
        random_state = self.random_state

        indices = np.arange(np.array(X).shape[0])
        split_list = []
        for i in range(0, self.iterations):
            x_train, x_test, y_train, y_test, idx1, idx2 = train_test_split(X,
                                                                            y,
                                                                            indices,
                                                                            test_size=test_size,
                                                                            stratify=(
                                                                                stratify if stratify is not None else None),
                                                                            random_state=(
                                                                                random_state + i if random_state else None))
            split_list.append([x_train, x_test, y_train, y_test, idx1, idx2])

        return split_list

    def fit_predict(self, split_list=None, correlation_data=None):
        self._reset_scores()
        self.correlation_data = correlation_data
        if self.SPEARMAN in self.metrics and self.correlation_data is None:
            raise ValueError(
                'If you want to use spearman correlation you need to pass the correlation data. Parameter: '
                'correlation_data')

    def _train(self, x_train=None, y_train=None):
        pass

    def _predict(self, x_test=None, y_test=None, idx2=None):
        pass

    def _scores(self, model_name, y_true=None, predictions=None, predict_probas=None, idx1=None, idx2=None):
        pass

    def _average_prec(y_true=None, _predictions=None, _predict_probas=None, _idx1=None, _idx2=None, _corr_data=None):
        labels = np.unique(y_true)
        res = []
        for label in labels:
            score = average_precision_score(y_true, _predict_probas[:, label], pos_label=label)
            res.append([str(label), score])
        return res

    def _roc_auc(y_true=None, _predictions=None, predict_probas=None, _idx1=None, _idx2=None, _corr_data=None):
        return roc_auc_score(y_true, predict_probas[:, 1])

    def _spearman(y_true=None, _predictions=None, predict_probas=None, idx1=None, idx2=None, correlation_data=None):
        pass

    def _classification_report(y_true=None, predictions=None, _predict_probas=None, _idx1=None, _idx2=None,
                               _corr_data=None):
        return classification_report(y_true, predictions, output_dict=True)

    def _avg_classification_reports(classification_reports):

        ACCURACY = 'accuracy'
        keys = classification_reports[0].keys()
        class_metrics = classification_reports[0][list(keys)[0]].keys()

        # create template for new report
        averaged_report = {}
        for key in keys:
            if key == ACCURACY:
                averaged_report[key] = []
                continue
            else:
                averaged_report[key] = {}
            for metric in class_metrics:
                averaged_report.get(key)[metric] = []

        for classification_report in classification_reports:
            for key in keys:

                if key == ACCURACY:
                    averaged_report.get(key).append(classification_report[key])
                    continue

                metrics_map = classification_report[key]

                for metric in class_metrics:
                    score = metrics_map[metric]
                    averaged_report.get(key).get(metric).append(score)

        for label, metrics_map in averaged_report.items():
            if label == ACCURACY:
                averaged_report[label] = mean(metrics_map)
                continue

            for metric, scores in metrics_map.items():
                metrics_map[metric] = mean(scores)

        return averaged_report

    def _is_valid_metrics(self, metrics):
        if metrics is None or not metrics:
            raise ValueError('You need to supply at least one metric using parameter metrics, '
                             'supported metrics are: ', self.SUPPORTED_METRICS)

        if not set(metrics).issubset(self.SUPPORTED_METRICS):
            raise ValueError('Supported metrics are: ', self.SUPPORTED_METRICS)

        return True

    def _calc_n_add_scores(self, scores_map=None, y_true=None, predictions=None, predict_probas=None, idx1=None,
                           idx2=None, corr_data=None):
        for metric in self.metrics:
            score = self.METRICS_FUNCTIONS[metric](y_true, predictions, predict_probas, idx1, idx2, corr_data)
            scores_map[metric].append(score)

    def _reset_scores(self):
        pass
