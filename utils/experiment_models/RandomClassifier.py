import numpy as np
from numpy.random import RandomState


class RandomClassifier:
    """
    RandomClassifier represents a random classifier with Sklearn functionality.

    Attributes
    ----------
    change_state : bool
        if true, the random seed for the generator will be changed for each call to predict()
        and predict_proba() (Default None)
    random_state : int
        used for reproducibility (Default None)

    Methods
    -------
    fit(_X=None,y=None)
        Sets the instance variable _classes to the unique class labels in y
    fit_predict(X=None,y=None)
        Sets the instance variable _classes to the unique class labels in y and then calls predict() with X
    predict(X=None)
        Returns X number of random predictions using numpy.random RandomState
    predict_proba(X=None)
        Returns X number of random pseudo probabilities.
    """

    def __init__(self, change_state=False, random_state=None):
        """
        Parameters
        ----------
        change_state : bool, optional
            Whether or not to change the random seed of the prediction generator (default is None)
        random_state : int, optional
            Used for reproducibility (default is None)

        Raises
        ------
        ValueError
            If change_state is set to True but random_state is set to None.
        """
        generator = RandomState(random_state)
        self._generator = generator
        self._seed = random_state

        if change_state and random_state is None:
            raise ValueError('change_state cannot be True if random_state is None. Please provide a seed.')

        self._change_state = change_state

    def fit(self, _X=None, y=None):
        """Sets the instance variable _classes to the unique class labels in y

        Parameters
        ----------
        _X : X{array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        """
        self._classes = np.unique(y)
        return

    def fit_predict(self, X=None, y=None):
        """Sets the instance variable _classes to the unique class labels in y and then calls predict() with X

        Parameters
        ----------
        X : X{array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        """
        self._classes = np.unique(y)
        return self.predict(X)

    def predict(self, X=None):
        """Returns X number of random predictions using numpy.random RandomState

        Parameters
        ----------
        X : X{array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.
        """
        predictions = np.array(self._generator.choice(a=self._classes, size=X.shape[0]))
        if self._change_state:
            self._change_seed()
        return predictions

    def predict_proba(self, X=None):
        """Returns X number of random pseudo probabilities.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.
        """
        sample_length = X.shape[0]

        probabilites = []

        for i in range(0, sample_length):
            predict_probas = []
            prediction = self._generator.choice(a=self._classes)

            for label in self._classes:
                if label == prediction:
                    predict_probas.append(1)
                else:
                    predict_probas.append(0)

            probabilites.append(predict_probas)

        if self._change_state:
            self._change_seed()

        return np.array(probabilites)

    def _change_seed(self):
        """Changes seed of prediction generator
        """
        self._seed = self._seed + 1
        self._generator.seed(self._seed)
