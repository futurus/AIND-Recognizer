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
    
    # repeat recognize for each word in test set
    for word_id in test_set.get_all_Xlengths().keys():
        # feature extraction
        X, lengths = test_set.get_all_Xlengths()[word_id]

        # dict holding {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... }
        probs = {}
        # hold max log likelihood for best guess
        maxLogL = float("-Inf")
        guess = None

        # try each model to find its probabilities and record max/guess
        for word, model in models.items():
            try:
                logL = model.score(X, lengths)
                probs[word] = logL

                if logL > maxLogL:
                    maxLogL = logL
                    guess = word

            except Exception as e:
                pass
                # print("Failure on word {}. Err: {}".format(word, str(e)))

        probabilities.append(probs)
        guesses.append(guess)

    return probabilities, guesses
