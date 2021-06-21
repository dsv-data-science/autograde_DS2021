"""The module preprocessing is used for preprocessing of exams.

    * get_text_word_lengths - Takes a corpus and returns the lengths of each text in words.

    * get_text_character_lengths - Takes a corpus and returns the lengths of each text in characters.

    * print_list_stats - Takes a list of ints and prints: mean, median, max, and min.

    * average_report - Takes a list of dict-like sklearn classification reports and averages them.

    * format_report - Takes a dict-like classification report in the format given by average_report() and formats
    it into a readable printable string.
"""

from statistics import mean, median


def get_text_word_lengths(corpus=None):
    """Takes a corpus and returns the lengths of each text in words.

    :param corpus: corpus.
    :return: a list of the lengths of each text in words.
    :rtype: list
    """
    text_lengths = []
    for text in corpus:
        text_lengths.append(len(text.split()))

    return text_lengths


def get_text_character_lengths(corpus):
    """Takes a corpus and returns the lengths of each text in characters.

    :param corpus: corpus.
    :return: a list of the lengths of each text in characters.
    :rtype: list
    """
    character_lengths = []
    for text in corpus:
        character_lengths.append(len(text))

    return character_lengths


def print_list_stats(stats=None):
    """Takes a list of integers and prints mean, median, max and min.

    :param stats: A list of ints.
    :type stats: list
    """
    print('Mean: {:0.0f}. Median: {:0.0f}. '
          'Longest: {:0.0f}. Shortest: {:0.0f}'.
          format(mean(stats), median(stats), max(stats), min(stats)))


def average_report(classification_reports):
    """Takes a list of dict-like sklearn classification reports
    (See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html), and returns
    the average of all metrics between the reports.

    :param classification_reports: list of dict-like sklearn classification reports.
    :return: a dict of averaged metrics between the reports.
    """
    keys = classification_reports[0].keys()
    class_metrics = classification_reports[0][list(keys)[0]].keys()

    # create template for new report
    averaged_report = {}
    for key in keys:
        if key == 'accuracy':
            averaged_report[key] = []
            continue
        else:
            averaged_report[key] = {}
        for metric in class_metrics:
            averaged_report.get(key)[metric] = []

    # add values to template
    for classification_report in classification_reports:
        for key in keys:

            if key == 'accuracy':
                averaged_report.get(key).append(classification_report[key])
                continue

            metrics_map = classification_report[key]

            for metric in class_metrics:
                score = metrics_map[metric]
                averaged_report.get(key).get(metric).append(score)

    # average values
    for label, metrics_map in averaged_report.items():
        if label == 'accuracy':
            averaged_report[label] = mean(metrics_map)
            continue

        for metric, scores in metrics_map.items():
            metrics_map[metric] = mean(scores)

    return averaged_report


def format_report(report):
    """Takes a dict-like classification report in the format given by average_report() and formats it into
    a readable printable string.

    :param report: dict-like classification report in the format given by average_report().
    :return: readable print string.
    """
    formated_report = ''
    for label, metrics in report.items():
        formated_report += '{}-> '.format(label)
        if type(metrics) == dict:
            for metric, score in metrics.items():
                formated_report += '{}: {:.2f}, '.format(metric, score)
        else:
            formated_report += '{:.2f}'.format(metrics)
        formated_report += '\n'

    return formated_report
