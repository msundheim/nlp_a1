# models.py

import numpy as np
import random

from sentiment_data import *
from utils import *

from collections import Counter

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """ TODO: add comments
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self) -> Indexer:
        """
        Get Indexer for this feature extractor object.
        """
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        # Process sentence and update featurizer and feature vector.
        feat_vect = Counter()
        for word in sentence:
            # Process all words as lowercase.
            lower_word = word.lower()

            # If add_to_indexer is True, grow dimensionality of Indexer (featurizer).
            if add_to_indexer:
                index = self.indexer.add_and_get_index(lower_word)
            else:
                index = self.indexer.index_of(lower_word)

            # With Counter (feature vector), count the frequency of each word in sentence.
            # If not add_to_indexer, throw out words that aren't present in featurizer.
            if index != -1:
                feat_vect[index] += 1

        # TODO: throw out stopwords?
        
        # Return feature vector.
        return feat_vect
        


class BigramFeatureExtractor(FeatureExtractor):
    """ TODO:
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class BetterFeatureExtractor(FeatureExtractor):
    """ TODO:
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """ TODO: add comments
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.weights = np.zeros(len(indexer)) # Initialize all weights as 0.

    def get_indexer(self) -> Indexer:
        return self.indexer
    
    def get_weights(self) -> np.array:
        return self.weights

    def set_weights(self, new_weights: np.array):
        self.weights = new_weights

    def predict(self, sentence: List[str]) -> int:
        #TODO: write

        # Get feature vector for given sentence.
        featurizer = UnigramFeatureExtractor(self.indexer)
        feat_vect = featurizer.extract_features(sentence, False)
        
        # Compute score of sentence (dot product of weight vector and feature vector frequencies).
        weights_rel = self.weights[list(feat_vect.keys())]
        score = np.dot(list(feat_vect.values()), weights_rel)

        # Check for correct prediction.
        return 1 if score > 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    """ TODO:
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self):
        raise Exception("Must be implemented")


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """ TODO: add comments and refactor
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    # Get vocabulary and all feature vectors.
    vects = list()
    for ex in train_exs:
        vects.append(feat_extractor.extract_features(ex.words, True))
        
    # Initialize new perceptron model.
    featurizer = feat_extractor.get_indexer()
    per_model = PerceptronClassifier(featurizer)

    # Learn from each example to train perceptron.
    # TODO: try with set number of epochs vs. convergence
    weights = per_model.get_weights()
    for t in range(0, 100): # TODO: adjust
        step_size = 1 / (t + 1)

        # Shuffle examples each epoch.
        random.seed(27)
        indices = list(range(0, len(train_exs)))
        random.shuffle(indices)

        # Update perceptron for each training example.
        for i in indices:
            # Get feature vector.
            feat_vect = vects[i]

            # Compute score of sentence (dot product of weight vector and feature vector frequencies).
            weights_rel = weights[list(feat_vect.keys())]
            score = np.dot(list(feat_vect.values()), weights_rel)

            # Check for correct prediction.
            y_pred = 1 if score > 0 else 0
            y_true = train_exs[i].label
            if y_pred == y_true:
                continue
            else:
                update_vals = step_size * np.array(list(feat_vect.values()))

                if y_true == 1:
                    # Need to increase weight.
                    weights_rel = np.add(weights_rel, update_vals)
                else:
                    # Need to decrease weight.
                    weights_rel = np.subtract(weights_rel, update_vals)
        
            # Update model with learned weights.
            weights[list(feat_vect.keys())] = weights_rel
        per_model.set_weights(weights)
    return per_model
        


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """ TODO:
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    raise Exception("Must be implemented")


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model