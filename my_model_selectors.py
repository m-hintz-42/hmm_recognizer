import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        min_model = None
        min_bic = float("inf")

        for state in range(self.min_n_components, self.max_n_components + 1):
            try:
                score, model = self.score_bic(state)
                if score < min_bic:
                    min_bic = score
                    min_model = model

            except Exception as e:
                pass

        if min_model:
            return min_model
        else:
            return self.base_model(self.n_constant)

    def score_bic(self, state):
        model = self.base_model(state)

        # Number of free parameters
        params = (state ** 2) + (2 * model.n_features * state) - 1

        logL = model.score(self.X, self.lengths)
        logN = np.log(len(self.X))

        BIC = -2 * logL + params * logN

        return BIC, model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

    DIC = log(P(X(i)) - 1/(M - 1) * sum(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        max_model = None
        max_dic = float("-inf")

        for state in range(self.min_n_components, self.max_n_components + 1):
            try:
                score, model = self.score_dic(state)
                if score > max_dic:
                    max_dic = score
                    max_model = model

            except Exception as e:
                pass

        if max_model:
            return max_model
        else:
            return self.base_model(self.n_constant)

    def score_dic(self, state):
        log_liklihoods =[]
        model = self.base_model(state)
        for word, (X, lengths) in  self.hwords.items():
            if word != self.this_word:
                log_word = model.score(X, lengths)
                log_liklihoods.append(log_word)

        DIC = model.score(self.X, self.lengths) - np.mean(log_liklihoods)
        return DIC, model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        max_model = None
        max_cv = float("-inf")

        for state in range(self.min_n_components, self.max_n_components + 1):
            try:
                if len(self.sequences) > 2:
                        score, model = self.score_cv(state)
                        if score > max_cv:
                            max_cv = score
                            max_model = model

            except Exception as e:
                pass

        if max_model:
            return max_model
        else:
            return self.base_model(self.n_constant)

    def score_cv(self, state):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        log_likelihood = []

        for train, test in KFold(n_splits=2).split(self.sequences):
            self.X, self.lengths = combine_sequences(train, self.sequences)
            test_x, test_lengths = combine_sequences(test, self.sequences)

            model = self.base_model(state)
            log_likelihood.append(model.score(test_x, test_lengths))

        return np.mean(log_likelihood), model
