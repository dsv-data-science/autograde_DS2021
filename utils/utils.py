"""The module preprocessing is used for preprocessing of exams.

    * fetch_freetext - Takes a list of exams and returns a dictionary of <Free-text question, [Responses, Grades,
    Max Grades]> key-value pairs.

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

"""

from collections import OrderedDict
import configs.configs as configs
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from statistics import mean, median
import random
from random import shuffle


def fetch_freetext(datasets=None):
    """Take a list of exams and returns a dictionary of <Free-text question, [Responses, Grades,
    Max Grades]> key-value pairs.

    :param datasets: The lists of datasets to fetch free-text questions from (default is None)
    :type datasets: Collection
    :return: a dictionary with free-text questions as keys and [responses, grades, max grades] as values.
    :rtypee: OrderedDict
    """

    if datasets is None:
        raise ValueError('Parameter need to be a list of datasets.')

    current_question = 1  # the first question of each exam always starts at 1
    question_answers = OrderedDict()  # the returning hashmap of questions, responses and grades.

    for exam in datasets:  # for each exam in datasets
        # get all columns with correct answers
        right_answer_cols = [col for col in exam.columns if configs.CORRECT_ANSWER_COLUMN in col]

        # for each right-answer column
        for right_answer_col in right_answer_cols:
            # get unique right answers from col after removing '-' unique set of values.
            unique_right_answers = set(exam[right_answer_col].unique()) - {'-'}
            # if column only had '-' values then it is a free-text question
            if len(unique_right_answers) == 0:
                # get question bank for the specific question corresponding to the right_answer_col number
                question_bank = exam[configs.QUESTION_COLUMN + str(current_question)].unique()
                # get the maximum grade possible for the current question_bank
                maximum_grade = \
                    [col for col in exam.columns if configs.SCORE_COLUMN + str(current_question) in col][0].split('/',
                                                                                                                  1)[-1]

                # for each question the question bank
                for question in question_bank:
                    index = exam.index  # indexes
                    # condition = all indices where question_column + question_nr are the same as question
                    condition = exam[configs.QUESTION_COLUMN + str(current_question)] == question
                    # get indices where condition is met
                    question_indices = index[condition].tolist()
                    # get answer and score columns and rows for specific question where indices match
                    scores_answers = exam.iloc[question_indices].filter(
                        regex=configs.ANSWER_COLUMN + str(current_question) + '|' + configs.SCORE_COLUMN + str(
                            current_question))
                    # remove empty response rows
                    scores_answers = scores_answers[
                        scores_answers[configs.ANSWER_COLUMN + str(current_question)] != '-']
                    # convert to list.
                    # E.g: [[Score1, Answer1], [Score2, Answer2], [Score3, Answer3], [Score4, Answer4] ..]
                    scores_answers = scores_answers.values.tolist()

                    # for each [Score_i, Answer_i] in scores_answers, add the maximum grade.
                    # E.g: [[Score1, Answer1, max grade], [Score2, Answer2, max grade],
                    # [Score3, Answer3, max grade], [Score4, Answer4, max grade] ..]
                    for score_answer in scores_answers:
                        score_answer.append(maximum_grade)

                    # the map uses the question as key, if key does not exist then val = None
                    val = question_answers.get(question)
                    if val is None:
                        # if key does not exist, create new key with question and put scores_answers as value
                        question_answers[question] = scores_answers
                    else:
                        # if key exist, then just add new scores_answers to val
                        question_answers[question] = val + scores_answers

            # finished with current right-answer column, increment current question and continue
            current_question = current_question + 1

        # finished with current exam, reset current_question and begin new exam
        current_question = 1

    return question_answers


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
