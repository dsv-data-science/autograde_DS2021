"""The module preprocessing is used for preprocessing of exams.

    * process_texts - Takes a list of texts for preprocessing and returns the
    preprocessed versions.

    * binary_sampling_questions - Takes a dataset and a range of question numbers and returns a random binary sampling
    of the responses for each question.

    * convert_sample_data - Converts the output of binary_sampling_questions into a dataframe and returns it.

    * get_random_responses - Takes a dataset with questions and returns n random responses of a given question.

    * get_top_n_words - Gets the top n most frequent word in a corpus by using CountVectorizer from SKlearn library.

    * get_text_word_lengths - Takes a corpus and returns the lengths of each text in words.

    * get_text_character_lengths - Takes a corpus and returns the lengths of each text in characters.

    * print_list_stats - Takes a list of ints and prints: mean, median, max, and min.

    * scramble_text - Takes a text and scrambles the words.

    * scramble_texts - Takes a list of texts and scrambles the words in each text.

    * average_report - Takes a list of dict-like sklearn classification reports and averages them.

    * format_report - Takes a dict-like classification report in the format given by average_report() and formats
    it into a readable printable string.
"""

import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from statistics import mean, median
import random
from random import shuffle


def process_texts(texts=None, punctuations=True, numbers=True, single_character=True):
    """ Takes a list of texts and returns the preprocessed versions.

    :param texts: The texts to preprocess
    :type texts: list
    :param punctuations: If punctuations should be removed
    :type punctuations: bool
    :param numbers: If numbers should be removed
    :type numbers: bool
    :param single_character: If single characters should be removed
    :type single_character: bool
    :return: The preprocessed versions of texts.
    :rtype: list
    """

    if texts is None:
        return

    # removes tags from the text and returns the new version.
    def preprocess_tags(text):
        return tag_re.sub('', text)

    # removes punctuations from the text and returns the new version.
    def preprocess_punctuations(text):
        return re.sub(r'[^\w\s]', '', text)

    # removes numbers from the text and returns the new version.
    def preprocess_numbers(text):
        return re.sub(r'\d+', '', text)

    # removes single characters from the text and returns the new version.
    def preprocess_characters(text):
        return re.sub(r"\s+[a-zA-Z]\s+", ' ', text)

    # removes multiple spaces from the text and returns the new version.
    def preprocess_multiple_spaces(text):
        return re.sub(r'\s+', ' ', text)

    tag_re = re.compile(r'<[^>]+>')
    new_texts = []
    for text in texts:
        text = preprocess_multiple_spaces(text)
        text = preprocess_tags(text)
        if punctuations:
            text = preprocess_punctuations(text)
        if numbers:
            text = preprocess_numbers(text)
        if single_character:
            text = preprocess_characters(text)
        new_texts.append(text)

    return new_texts


def binary_sampling_questions(dataset=None, question_nr_column='', class_column='', target_column='',
                              question_nr_range=[1, 2], n_samples=10, replace=False, random_state=None):
    """ Takes a dataset and a range of question numbers and returns a random binary sampling of the responses for each
    question.

    :param dataset: The dataset to fetch questions from.
    :type dataset: DataFrame
    :param question_nr_column: Column name of the question number column.
    :type question_nr_column: String
    :param class_column: Column name of the class column.
    :type class_column: String
    :param target_column: Column name of which column should be sampled.
    :type target_column: String
    :param question_nr_range: The range of which questions should be sampled. [x, y], from x to but excluding y.
    :type question_nr_range: List
    :param n_samples: The number of samples per question.
    :type question_nr_range: int
    :param replace: Whether DataFrame.sample() should use replace. Allows or disallows sampling of the same row more than once.
    :type replace: bool
    :param random_state: Random state.
    :type random_state: int
    :return: A list of samples as: [[Question_nr, {0:[negative samples], 1:[positivesamples]}], [Question_nr, ..]..]
    :rtype: List
    """

    if dataset is None:
        raise ValueError('Dataset cannot be None.')

    samples = []
    # for each question nr in range question_nr_range
    for question_nr in range(question_nr_range[0], question_nr_range[1]):
        # sample n_samples negative samples from the dataset
        negative_samples = dataset.loc[
            (dataset[question_nr_column] == question_nr)
            &
            (dataset[class_column] == 0)
            ][target_column].sample(n=n_samples, replace=replace, random_state=random_state)

        # sample n_samples positive samples from the dataset
        positive_samples = dataset.loc[
            (dataset[question_nr_column] == question_nr)
            &
            (dataset[class_column] == 1)
            ][target_column].sample(n=n_samples, replace=replace, random_state=random_state)

        # append the samples and the question nr to the list of samples as:
        # [Question_Nr, {0:Negative samples, 1:Positive samples}]
        samples.append(['Question Number: {}'.format(question_nr), {0: negative_samples, 1: positive_samples}])

    return samples


def convert_sample_data(samples=None):
    """ Converts output of binary_sampling_questions function into a dataframe.

    :param samples: List of samples in the format given by function: binary_sampling_questions.
    :return: A dataframe of samples.
    """

    sample_df = pd.DataFrame(columns=['Question_Nr', 'Low Grade', 'High Grade'])

    # for each question_nr and its map in samples
    for question_nr, mapped_samples in samples:
        # get negative samples from map
        negative_samples = list(mapped_samples[0])
        # get positive samples from map
        positive_samples = list(mapped_samples[1])
        # As len(negative_samples) == len(positive_samples), we can iterate through range of one of them
        for i in range(0, len(negative_samples)):
            row = {'Question_Nr': question_nr, 'Low Grade': negative_samples[i], 'High Grade': positive_samples[i]}
            sample_df = sample_df.append(row, ignore_index=True)

    return sample_df


def get_random_responses(dataset=None, question_nr_column=None, response_column=None, question_nr=None,
                         class_column=None,
                         class_label=None, class_labels=None, n_responses=10, random_state=None):
    """ Takes a dataset with questions and returns n random responses of question: question_nr.

    :param dataset: dataset containing questions and responses
    :param question_nr_column: the column with question numbers
    :type question_nr_column: str
    :param response_column: the column with responses
    :type response_column: str
    :param question_nr: the question number
    :type question_nr: int
    :param class_column: the column with classes
    :type class_column: str
    :param class_label: which class label the responses should correspond to
    :param class_labels: if you want responses from multiple class labels
    :type class_labels: iterable
    :param n_responses: the amount of responses you want. if class_labels is not None, the amount
    of responses are per class label
    :type n_responses: int
    :param random_state: random seed for reproducibility
    :return: a dictionary with class label - responses pairs if class_labels is not None.
    Returns a list of responses if class label is None.
    """

    if question_nr_column is None or response_column is None or class_column is None:
        raise ValueError('You need to pass the following columns: \n'
                         'question_nr_column, which is the column of question numbers in the dataset; \n'
                         'response_column, which is the column of responses in the dataset; \n'
                         'class_column, which is the column of class labels in the dataset.')

    if question_nr is None:
        raise ValueError('You need to pass a question number to fetch responses from. Parameter: question_nr.')

    def get_question_nr(dt):
        return dt.loc[
            dt[question_nr_column] == question_nr
            ]

    def get_class_label(dt, label):
        return dt.loc[
            dt[class_column] == label
            ]

    def get_responses(dt, class_label):
        return get_class_label(dt, class_label)[[response_column, class_column]].sample(n=n_responses,
                                                                                        random_state=random_state
                                                                                        ).to_records(index=False)

    question_nr_dt = get_question_nr(dataset)  # get sub-dataframe only containing question_nr
    if class_labels:  # if class labels were passed, get responses for each class label
        responses = {}
        for class_label in class_labels:
            responses[class_label] = get_responses(question_nr_dt, class_label)
        return responses
    elif class_label is not None:  # if only one class label were passed, get responses for that class label
        return get_responses(question_nr_dt, class_label)
    else:  # else get n random responses from any class at random
        return question_nr_dt[[response_column, class_column]].sample(n=n_responses,
                                                                      random_state=random_state
                                                                      ).to_records(index=False)


def get_top_n_words(texts=None, n_words=10, count_vectorizer=None):
    """Gets top n words in corpus by using CountVectorizer from Sklearn library.

    :param texts: Corpus.
    :param n_words: Amount of top n words to retrieve.
    :type n_words: int
    :param count_vectorizer: If you want to use your own CountVectorizer() from Sklearn library.
    :return: top n words in corpus.
    """
    if texts is None:
        return None

    if count_vectorizer is None:
        count_vectorizer = CountVectorizer()

    vect = count_vectorizer.fit(texts)
    bag_of_words = vect.transform(texts)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vect.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n_words]


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


def scramble_text(sentence, random_state=None):
    """Takes a text and scrambles the words.

    :param sentence: The text to scramble.
    :type sentence: String
    :param random_state: Used for reproducibility.
    :type random_state: int
    :return: scrambled text
    :rtype String
    """
    words = sentence.split()
    random.Random(random_state).shuffle(words) if random_state else shuffle(words)
    return ' '.join(words)


def scramble_texts(texts=None, random_state=None):
    """Takes a list of texts and scrambles the words in each text.

    :param texts: list of texts to scramble.
    :param random_state: Used for reproducibility.
    :type random_state: int
    :return: list of new scrambled texts.
    """

    scrambled_data = []
    for sentence in texts:
        new_sentence = scramble_text(sentence, random_state)
        scrambled_data.append(new_sentence)
    return scrambled_data


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
