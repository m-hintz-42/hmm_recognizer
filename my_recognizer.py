import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for test_X, test_lengths in test_set.get_all_Xlengths().values():

        final_score = float("-inf")
        final_guess = ''
        log_liklihoods = {}

        for test_word, model in models.items():
            try:
                test_score = model.score(test_X, test_lengths)
                log_liklihoods[test_word] = test_score

            except Exception as e:
                log_liklihoods[test_word] = float("-inf")
                continue

            if test_score > final_score:
                final_score = test_score
                final_guess = test_word

        probabilities.append(log_liklihoods)
        guesses.append(final_guess)

    return probabilities, guesses
