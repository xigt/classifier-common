import gzip
import logging
import os
import pickle
import multiprocessing

from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import numpy as np
from sklearn.linear_model import LogisticRegression
from abc import abstractclassmethod

LOG = logging.getLogger()


class DataInstance(object):
    def __init__(self, label, feats):
        self.label = label
        self.feats = feats

class DocInstance(DataInstance):
    """
    Wrapper class to hold the path, doc_id, label, and features
    """

    def __init__(self, doc_id, label, feats, path):
        super().__init__(label, feats)
        self.path = path
        self.doc_id = doc_id

class StringInstance(DataInstance):
    def __init__(self, id, label, feats):
        super().__init__(label, feats)
        self.id = id

class Distribution(object):
    def __init__(self, classes, probs):
        self.dict = {}
        self.best_class = None
        self.best_prob = 0.0
        for c, p in zip(classes, probs):
            self.dict[c] = p
            if p > self.best_prob:
                self.best_class = c
                self.best_prob = p

    def get(self, key, default=None):
        return self.dict.get(key, None)

    def classes(self): return self.dict.keys()



class ClassifierWrapper(object):
    """
    This class implements a wrapper class to combine sklearn's
    learner, vectorizer, and feature selector classes into one
    serializable object.
    """
    def __init__(self):
        self.dv = DictVectorizer(dtype=int)
        self.feat_selector = None
        self.learner = None

    def _vectorize(self, data, testing=False):
        if testing:
            return self.dv.transform(data)
        else:
            return self.dv.fit_transform(data)

    def _vectorize_and_select(self, data, labels, num_feats=None, testing=False):

        # Start by vectorizing the data.
        vec = self._vectorize(data, testing=testing)

        # Next, filter the data if in testing mode, according
        # to whatever feature selector was defined during
        # training.
        if testing:
            if self.feat_selector is not None:
                # LOG.info('Feature selection was enabled during training, limiting to {} features.'.format(
                #     self.feat_selector.k))
                return self.feat_selector.transform(vec)
            else:
                return vec

        # Only do feature selection if num_feats is positive, and there are more features
        # than max_num
        elif num_feats is not None and (num_feats > 0) and num_feats < vec.shape[1]:
            LOG.info('Feature selection enabled, limiting to {} features.'.format(num_feats))
            self.feat_selector = SelectKBest(chi2, num_feats)
            return self.feat_selector.fit_transform(vec, labels)

        else:
            LOG.info("Feature selection disabled, all available features are used.")
            return vec

    def _checklearner(self):
        if self.learner is None:
            raise Exception("Learner must be specified.")

    def train(self, data, num_feats=None, weight_path=None):
        """
        :type data: list[DataInstance]
        """
        self._checklearner()
        labels = [d.label for d in data]
        feats = [d.feats for d in data]

        vec = self._vectorize_and_select(feats, labels, num_feats=num_feats, testing=False)
        self.learner.fit(vec, labels)
        if weight_path is not None:
            LOG.info('Writing feature weights to "{}"'.format(weight_path))
            self.dump_weights(weight_path)

    def test(self, data, prev_label_func=None):
        """
        Given a list of document instances, return a list
        of the probabilities of the Positive, Negative examples.

        :type data: Iterable[DataInstance]
        :rtype: list[Distribution]

        """
        self._checklearner()
        labels = []
        feats = []

        # We need to make this loop happen this way in case
        # the data is a generator, and doing list
        # comprehensions will result in one list being empty.
        prev_class = None

        for datum in data:
            # If the "use_prev_label" is true, use it.
            if prev_label_func is not None and prev_class is not None:
                prev_label_feat = prev_label_func(prev_class)
                datum.feats[prev_label_feat] = True

            vec = self._vectorize_and_select([datum.feats], [datum.label], testing=True)
            probs = self.learner.predict_proba(vec)

            d = Distribution(self.classes(), probs[0])
            prev_class = d.best_class

            yield d


    def classes(self):
        self._checklearner()
        return self.learner.classes_.tolist()

    def feat_names(self):
        return np.array(self.dv.get_feature_names())

    def feat_supports(self):
        if self.feat_selector is not None:
            return self.feat_selector.get_support()
        else:
            return np.ones((len(self.dv.get_feature_names())), dtype=bool)

    def weights(self):
        """
        Get a list of features and their importances,
        either for logistic regression or adaboost.

        :return:
        """
        if isinstance(self.learner, AdaBoostClassifier):
            feat_weights = self.learner.feature_importances_
            return {f: feat_weights[j] for j, f in enumerate(self.feat_names()[self.feat_supports()])
                    if feat_weights[j] != 0}
        elif isinstance(self.learner, LogisticRegression):
            return {f: self.learner.coef_[0][j] for j, f in enumerate(self.feat_names()[self.feat_supports()])}

    def dump_weights(self, path, n=-1):
        with open(path, 'w') as f:
            sorted_weights = sorted(self.weights().items(), reverse=True, key=lambda x: x[1])
            for feat_name, weight in sorted_weights[:n]:
                f.write('{}\t{}\n'.format(feat_name, weight))

    def save(self, path):
        """
        Serialize the classifier out to a file.
        """
        if os.path.dirname(path): os.makedirs(os.path.dirname(path), exist_ok=True)
        f = gzip.GzipFile(path, 'w')
        pickle.dump(self, f)
        f.close()

    @classmethod
    def load(cls, path):
        f = gzip.GzipFile(path, 'r')
        c = pickle.load(f)
        assert isinstance(c, ClassifierWrapper)
        return c


class LogisticRegressionWrapper(ClassifierWrapper):
    def __init__(self):
        super().__init__()
        self.learner = LogisticRegression()

class AdaboostWrapper(ClassifierWrapper):
    def __init__(self):
        super().__init__()
        self.learner = AdaBoostClassifier()


def show_weights(cw, n=-1):
    # Group Features by class
    default_weights = {}

    if isinstance(cw.learner, LogisticRegression):
        classes = cw.learner.classes_
        defaults = cw.learner.intercept_
        coefs = cw.learner.coef_

        weights_per_class = -1 if n < 0 else n / len(classes)

        fmt = '{:<20s}{:.4g}'

        for i, c in enumerate(classes):
            d = defaults[i]
            print('- '*5 + c + ' -' * 5)
            print(fmt.format('<DEFAULT>', d))

            feat_weights = {f: coefs[i][j] for j, f in enumerate(cw.feat_names()[cw.feat_supports()])}.items()
            sorted_weights = sorted(feat_weights, reverse=True, key=lambda x: x[1])

            n = 0
            for feat, weight in sorted_weights:
                print(fmt.format(feat, weight))
                n+=1
                if weights_per_class != -1 and n >= weights_per_class:
                    break


    sys.exit()