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

    def calc_score(self, n):
        model = self.base_model(n)
        
        # From GaussianHMM doc:
        # params : string, optional 
        # Controls which parameters are updated in the training process. 
        # Can contain any combination of ‘s’ for startprob, ‘t’ for transmat, 
        # ‘m’ for means and ‘c’ for covars. Defaults to all parameters.
        p = model.startprob_.size + model.transmat_.size + model.means_.size + model.covars_.diagonal().size
        
        logL = model.score(self.X, self.lengths)
        logN = np.log(self.X.shape[0])
        score = -2 * logL + p * logN

        return model, score

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float("inf")
        best_model = self.base_model(self.n_constant)

        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                model, score = self.calc_score(n_components)

                if score < best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                if self.verbose:
                    print("Failure on {} with {} states. Err: {}".format(self.this_word, n_components, str(e)))

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def calc_dic(self, n_components):
        model = self.base_model(n_components)
        mean_score = np.mean([model.score(self.hwords[w][0], self.hwords[w][1]) for w in self.words if w != self.this_word])
        dic_score = model.score(self.X, self.lengths) - mean_score
		
        return model, dic_score

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float('-Inf')
        best_model = self.base_model(self.n_constant)
        
        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                model, score = self.calc_dic(n_components)
				
                if score >= best_score:
                    best_score = score
                    best_model = model

            except Exception as e:
                if self.verbose:
                    print("Failure on {} with {} states. Err: {}".format(self.this_word, n_components, str(e)))

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def calc_cv(self, n_splits=2):
        scores = []
        splitter = KFold(n_splits)

        for train, test in splitter.split(self.sequences):
            try:
                # Split according to splitter
                X_train, len_train = combine_sequences(train, self.sequences)
                X_test, len_test = combine_sequences(test, self.sequences)
                
                # train model
                hmm_model = GaussianHMM(n_components=n_splits, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(X_train, len_train)
                # evaluate using test data
                scores.append(hmm_model.score(X_test, len_test))

            except Exception as e:
                if self.verbose:
                    print("Failure on {} with {} states. Err: {}".format(self.this_word, n_splits, str(e)))

        return hmm_model, np.mean(scores)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float('-Inf')
        best_n = self.min_n_components
		
        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:			
                model, score = self.calc_cv(n_components)

                if score > best_score:
                    best_score = score
                    best_n = n_components

            except Exception as e:
                if self.verbose:
                    print("Failure on {} with {} states. Err: {}".format(self.this_word, n_components, str(e)))

        return self.base_model(best_n)
