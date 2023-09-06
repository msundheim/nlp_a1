# models.py

# Modules imported by Megan Sundheim.
import numpy as np
"""
# For Q6 plotting.
import matplotlib.pyplot as plt
"""

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
    """ 
    Extracts unigram bag-of-words features from a sentence.

    Relies on active feature extraction of training vocabulary prior to model training.
    
    Adds words to indexer in lowercase form.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer # Bijectively maps lowercase words to integral indices.

    def get_indexer(self) -> Indexer:
        """
        Get Indexer for this feature extractor object.
        """
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extracts feature vector (word frequencies) for given sentence.
        
        If add_to_indexer, add words to featurizer indexer.
        Else, discard words not in featurizer.
        """
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
        
        # Return feature vector.
        return feat_vect
        


class BigramFeatureExtractor(FeatureExtractor):
    """ 
    Bigram feature extractor analogous to the unigram one.
    
    Relies on active feature extraction of training vocabulary prior to model training.
    
    Adds bigrams to indexer in lowercase form.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self) -> Indexer:
        """
        Get Indexer for this feature extractor object.
        """
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extracts feature vector (bigram frequencies) for given sentence.
        
        If add_to_indexer, add bigrams to featurizer indexer.
        Else, discard bigrams not in featurizer.
        """
        # Process sentence and update featurizer and feature vector.
        feat_vect = Counter()
        for i in range(0, len(sentence) - 1):
            # Process all words as lowercase.
            word_1 = sentence[i].lower()
            word_2 = sentence[i + 1].lower()
            bigram = (word_1, word_2)

            # If add_to_indexer is True, grow dimensionality of Indexer (featurizer).
            if add_to_indexer:
                index = self.indexer.add_and_get_index(bigram)
            else:
                index = self.indexer.index_of(bigram)

            # With Counter (feature vector), count the frequency of each bigram in sentence.
            # If not add_to_indexer, throw out bigrams that aren't present in featurizer.
            if index != -1:
                feat_vect[index] += 1
        
        # Return feature vector.
        return feat_vect


class BetterFeatureExtractor(FeatureExtractor):
    """ 
    Better feature extractor...try whatever you can think of!
    
    Commented out code belongs to my Q8-9 feature modification: term frequency - 
    inverse document frequency (tf-idf).
    
    Contains a wrapper of the UnigramFeatureExtractor, as this feature extractor
    performed best among unigram, bigram, and tf-idf feature extractors.
    """
    def __init__(self, indexer: Indexer):
        self.unigram = UnigramFeatureExtractor(indexer)
        """
        # Fields for td-idf feature extractor.
        self.indexer = indexer
        self.word_in_sent = Counter()
        self.num_sentences = 0
        self.idf_table = dict()
        """

    def get_indexer(self) -> Indexer:
        """
        Get Indexer for this feature extractor object.
        """
        return self.unigram.get_indexer()

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extracts feature vector (word frequencies) for given sentence.
        
        If add_to_indexer, add words to featurizer indexer.
        Else, discard words not in featurizer.
        """
        # Return feature vector from unigram feature extractor.
        return self.unigram.extract_features(sentence, add_to_indexer)

    """
    # Functions belonging to tf-idf feature extractor.
    def calc_tf(self, sentence: List[str]) -> dict:
        # Calculate the term frequency for each word in a given sentence.
        word_freq = Counter()
        word_tf = dict()

        # Calculate word frequency for given sentence.
        for word in sentence:
            # Process all words as lowercase.
            lower_word = word.lower()
            index = self.indexer.index_of(lower_word)
            
            if index != -1:
                word_freq[index] += 1

                # Update term frequency for word.
                tf = word_freq[index] / len(sentence)
                word_tf[index] = tf

        return word_tf

    def calc_idf(self, train_exs: List[SentimentExample]):
        # Calculate inverse document frequency for each word in training examples.
        # This function only needs to be called once after featurizer filled.
        self.num_sentences = len(train_exs)
        for ex in train_exs:
            for word in set(ex.words):
                # Process all words as lowercase.
                lower_word = word.lower()
                index = self.indexer.index_of(lower_word)

                if index != -1:
                    # Update inverse word frequency for word.
                    self.word_in_sent[index] += 1
                    idf = np.log(self.num_sentences / self.word_in_sent[index])
                    self.idf_table[index] = idf

    def calc_tf_idf(self, tf_table: dict) -> dict:
        # Calculate the term frequency - inverse document frequency of each word in a sentence.
        tf_idf_table = dict()
        for index in tf_table.keys():
            tf = tf_table[index]
            idf = self.idf_table[index]
            tf_idf_table[index] = tf * idf

        return tf_idf_table
    """


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
    """
    Perceptron model that classifies positive/negative sentence sentiment.
    """
    def __init__(self, feat_extractor: FeatureExtractor):
        self.indexer = feat_extractor.get_indexer() # Training set vocabulary.
        self.weights = np.zeros(len(self.indexer)) # Initialize all weights as 0.
        self.feat_extractor = feat_extractor # Gets feature vector for a given sentence.

    def get_indexer(self) -> Indexer:
        """
        Get Indexer for this perceptron model.
        """
        return self.indexer
    
    def get_weights(self) -> np.array:
        """
        Get weights for this perceptron model.
        """
        return self.weights

    def set_weights(self, new_weights: np.array):
        """
        Update weights for this perceptron model.
        """
        self.weights = new_weights

    def print_weights(self):
        """
        For Q3, print 10 words with highest positive weight and 10 words with the
        lowest negative weight.
        """
        indices = list(self.indexer.ints_to_objs.keys())
        top_pos = np.array(indices[:10])
        top_neg = np.array(indices[:10])

        # Maintain minimum weight in highest weight array.
        min_w = min(self.weights[top_pos])
        min_idx = np.argmin(self.weights[top_pos])

        # Maintain maximum weight in lowest weight array.
        max_w = max(self.weights[top_neg])
        max_idx = np.argmax(self.weights[top_neg])

        # Get top 10 and bottom 10 weights.
        for index in indices:
            weight = self.weights[index]
            if weight > min_w:
                # Update top 10 list.
                top_pos[min_idx] = index
                min_w = min(self.weights[top_pos])
                min_idx = np.argmin(self.weights[top_pos])
            
            if weight < max_w:
                top_neg[max_idx] = index
                max_w = max(self.weights[top_neg])
                max_idx = np.argmax(self.weights[top_neg])
        
        # Sort words and weights.
        top_words = [(self.weights[index], self.indexer.get_object(index)) for index in top_pos]
        top_words.sort(reverse=True)
        bottom_words = [(self.weights[index], self.indexer.get_object(index)) for index in top_neg]
        bottom_words.sort()

        # Print top 10 and bottom 10 words and weights.
        print("\nHighest positive weights:")
        for pair in top_words:
            weight, word = pair
            print(f'{word}: {weight}')

        print("\nLowest negative weights:")
        for pair in bottom_words:
            weight, word = pair
            print(f'{word}: {weight}')
        print()

    def predict(self, sentence: List[str]) -> int:
        """
        Predicts the (binary) sentiment class of the given sentence.
        """
        # Get feature vector for given sentence.
        feat_vect = self.feat_extractor.extract_features(sentence, False)

        """
        # Calculates term frequency - inverse document frequency (tf-idf) (for BetterFeatureExtractor).
        if isinstance(self.feat_extractor, BetterFeatureExtractor):
            # Calculate term frequency and tf-idf for given sentence.
            tf_table = self.feat_extractor.calc_tf(sentence)
            tf_idf_table = self.feat_extractor.calc_tf_idf(tf_table)
            
            # Use tf-idf table as feature vector instead of sentence word frequency.
            new_vect = Counter()
            for index in feat_vect.keys():
                new_vect[index] = tf_idf_table[index]
            feat_vect = new_vect
        """
        
        # Compute score of sentence (dot product of weight vector and feature vector frequencies).
        weights_rel = self.weights[list(feat_vect.keys())]
        score = np.dot(list(feat_vect.values()), weights_rel)

        # Classify sentence as positive or negative given score.
        return 1 if score > 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Logistic regression model that classifies positive/negative sentence sentiment.
    """
    def __init__(self, feat_extractor: FeatureExtractor):
        self.indexer = feat_extractor.get_indexer() # Training set vocabulary.
        self.weights = np.zeros(len(self.indexer)) # Initialize all weights as 0.
        self.feat_extractor = feat_extractor # Gets feature vector for a given sentence.

    def get_indexer(self) -> Indexer:
        """
        Get Indexer for this logistic regression model.
        """
        return self.indexer
    
    def get_weights(self) -> np.array:
        """
        Get weights for this logistic regression model.
        """
        return self.weights

    def set_weights(self, new_weights: np.array):
        """
        Update Indexer for this logistic regression model.
        """
        self.weights = new_weights

    def predict(self, sentence: List[str]) -> int:
        """
        Predicts the (binary) sentiment class of the given sentence.
        """
        # Get feature vector for given sentence.
        feat_vect = self.feat_extractor.extract_features(sentence, False)

        """
        # Calculates term frequency - inverse document frequency (tf-idf) (for BetterFeatureExtractor).
        if isinstance(self.feat_extractor, BetterFeatureExtractor):
            # Calculate term frequency and tf-idf for given sentence.
            tf_table = self.feat_extractor.calc_tf(sentence)
            tf_idf_table = self.feat_extractor.calc_tf_idf(tf_table)
            
            # Use tf-idf table as feature vector instead of sentence word frequency.
            new_vect = Counter()
            for index in feat_vect.keys():
                new_vect[index] = tf_idf_table[index]
            feat_vect = new_vect
        """
        
        # Compute probability of sentence for positive class.
        weights_rel = self.weights[list(feat_vect.keys())]
        dot_product = np.dot(list(feat_vect.values()), weights_rel)
        e_exp = np.exp(dot_product)
        pos_prob = (e_exp) / (1 + e_exp)
        
        # Predict sentiment of given sentence given probability of positive sentiment.
        return 1 if pos_prob > 0.5 else 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """ 
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    # Actively extract features from training set and get all feature vectors for training set.
    vects = list()
    for ex in train_exs:
        vects.append(feat_extractor.extract_features(ex.words, True))
    
    """
    # Use tf-idf as feature vector instead of sentence word frequency (for BetterFeatureExtractor).
    if isinstance(feat_extractor, BetterFeatureExtractor):
        # Calculate inverse document frequency for training set featurizer.
        feat_extractor.calc_idf(train_exs)

        # For each sentence, calculate term frequency and tf-idf.
        for i in range(0, len(train_exs)):
            tf_table = feat_extractor.calc_tf(train_exs[i].words)
            tf_idf_table = feat_extractor.calc_tf_idf(tf_table)

            # Get feature vector.
            feat_vect = vects[i]
            
            # Use tf-idf instead of sentence word frequency.
            new_vect = Counter()
            for index in feat_vect.keys():
                new_vect[index] = tf_idf_table[index]
            vects[i] = new_vect
    """
        
    # Initialize new perceptron model.
    featurizer = feat_extractor.get_indexer()
    per_model = PerceptronClassifier(feat_extractor)

    # Learn from each example to train perceptron.
    weights = per_model.get_weights()
    for t in range(0, 25):
        step_size = 1 / (t + 1) # Dynamic step size.
        """
        # Constant step size (used for Q2).
        step_size = 1
        """

        # Shuffle examples each epoch (with random seed).
        np.random.seed(27)
        indices = list(range(0, len(train_exs)))
        np.random.shuffle(indices)

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
                # No update necessary.
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
    
    """
    # Print words with top 10 positive weights and bottom 10 negative weights.
    per_model.print_weights()
    """
    
    return per_model
        

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, epochs: int=25) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # Get vocabulary and all feature vectors.
    vects = list()
    for ex in train_exs:
        vects.append(feat_extractor.extract_features(ex.words, True))

    """
    # Use tf-idf as feature vector instead of sentence word frequency (for BetterFeatureExtractor).
    if isinstance(feat_extractor, BetterFeatureExtractor):
        # Calculate inverse document frequency for training set featurizer.
        feat_extractor.calc_idf(train_exs)

        # For each sentence, calculate term frequency and tf-idf.
        for i in range(0, len(train_exs)):
            tf_table = feat_extractor.calc_tf(train_exs[i].words)
            tf_idf_table = feat_extractor.calc_tf_idf(tf_table)

            # Get feature vector.
            feat_vect = vects[i]
            
            # Use tf-idf instead of sentence word frequency.
            new_vect = Counter()
            for index in feat_vect.keys():
                new_vect[index] = tf_idf_table[index]
            vects[i] = new_vect
    """
    # Initialize new logistic regression model.
    featurizer = feat_extractor.get_indexer()
    log_model = LogisticRegressionClassifier(feat_extractor)

    # Learn from each example to train logistic regression model.
    weights = log_model.get_weights()
    for t in range(0, epochs):
        step_size = 1 / (t + 1)

        # Shuffle examples each epoch (with random seed).
        np.random.seed(27)
        indices = list(range(0, len(train_exs)))
        np.random.shuffle(indices)

        # Update weights for each training example.
        for i in indices:
            # Get feature vector.
            feat_vect = vects[i]

            # Compute probability of sentence for positive class.
            weights_rel = weights[list(feat_vect.keys())]
            dot_product = np.dot(list(feat_vect.values()), weights_rel)
            e_exp = np.exp(dot_product)
            pos_prob = (e_exp) / (1 + e_exp)

            # Update weights based on true class label.
            y_true = train_exs[i].label
            update_feats = step_size * np.array(list(feat_vect.values()))

            if y_true == 1:
                # Need to increase weight.
                update_vals = update_feats * (1 - pos_prob)
                weights_rel = np.add(weights_rel, update_vals)
            else:
                # Need to decrease weight.
                update_vals = update_feats * pos_prob
                weights_rel = np.subtract(weights_rel, update_vals)
            
            # Update model with learned weights.
            weights[list(feat_vect.keys())] = weights_rel
        log_model.set_weights(weights)
    return log_model


"""
def train_log_reg_plot(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], feat_extractor: FeatureExtractor, epochs: int=25) -> LogisticRegressionClassifier:
    
    # For Q6, plot the training objective, dataset log likelihood, and dev accuracy vs. 
    # number of training iterations for several step sizes.

    # Initialize subplots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_title("Number of Training Iterations vs. Dataset Log Likelihood")
    ax1.set_xlabel("Number of Training Iterations")
    ax1.set_ylabel("Dataset Log Likelihood")
    ax2.set_title("Number of Training Iterations vs. Dev Accuracy")
    ax2.set_xlabel("Number of Training Iterations")
    ax2.set_ylabel("Dev Accuracy")

    # Get vocabulary and all feature vectors.
    vects = list()
    for ex in train_exs:
        vects.append(feat_extractor.extract_features(ex.words, True))
    featurizer = feat_extractor.get_indexer()

    # Get feature vectors for dev set.
    dev_vects = list()
    for ex in dev_exs:
        dev_vects.append(feat_extractor.extract_features(ex.words, False))

    step_sizes = [0.25, 0.5, 1]
    colors = ['red', 'blue', 'green', 'black']
    for j in range(0, len(step_sizes) + 1):
        if j < len(step_sizes):
            step_size = step_sizes[j]
        color = colors[j]

        # Initialize structures to hold results for Q6 plotting.
        log_likelihood = list()
        dev_accuracy = list()
        epoch_list = list()

        # Initialize new logistic regression model.
        log_model = LogisticRegressionClassifier(feat_extractor)

        # Learn from each example to train logistic regression model.
        weights = log_model.get_weights()
        for t in range(0, epochs):
            if j == len(step_sizes):
                step_size = 1 / (t + 1)
            epoch_list.append(t + 1)

            # Shuffle examples each epoch (with random seed).
            np.random.seed(27)
            indices = list(range(0, len(train_exs)))
            np.random.shuffle(indices)

            # Update weights for each training example.
            for i in indices:
                # Get feature vector.
                feat_vect = vects[i]

                # Compute probability of sentence for positive class.
                weights_rel = weights[list(feat_vect.keys())]
                dot_product = np.dot(list(feat_vect.values()), weights_rel)
                e_exp = np.exp(dot_product)
                pos_prob = (e_exp) / (1 + e_exp)

                # Update weights based on true class label.
                y_true = train_exs[i].label
                update_feats = step_size * np.array(list(feat_vect.values()))

                if y_true == 1:
                    # Need to increase weight.
                    update_vals = update_feats * (1 - pos_prob)
                    weights_rel = np.add(weights_rel, update_vals)
                else:
                    # Need to decrease weight.
                    update_vals = update_feats * pos_prob
                    weights_rel = np.subtract(weights_rel, update_vals)
                
                # Update model with learned weights.
                weights[list(feat_vect.keys())] = weights_rel
            log_model.set_weights(weights)

            # Calculate dataset log likelihood.
            curr_ll = 0
            for i in indices:
                # Get feature vector.
                feat_vect = vects[i]

                # Compute probability of sentence for true class.
                weights_rel = log_model.get_weights()[list(feat_vect.keys())]
                dot_product = np.dot(list(feat_vect.values()), weights_rel)
                e_exp = np.exp(dot_product)
                y_true = train_exs[i].label
                if y_true == 1:
                    # Positive class.
                    prob = (e_exp) / (1 + e_exp)
                else:
                    # Negative class.
                    prob = 1 - ((e_exp) / (1 + e_exp))

                # Update dataset log likelihood.
                curr_ll += np.log(prob)
            log_likelihood.append(curr_ll)

            # Calculate dev accuracy.
            dev_correct = 0
            for ex in dev_exs:
                # Predict class of dev set example.
                y_pred = log_model.predict(ex.words)
                y_true = ex.label
                if y_pred == y_true:
                    dev_correct += 1

            # Calculate dev accuracy.
            curr_acc = dev_correct / len(dev_exs)
            dev_accuracy.append(curr_acc)
    
        # Print results for current step size.
        if j == len(step_sizes):
            step_size = '1 / t'
        ax1.plot(epoch_list, log_likelihood, color=color, label=step_size)
        ax1.legend(title='Step Size')
        ax2.plot(epoch_list, dev_accuracy, color=color, label=step_size)
        ax2.legend(title='Step Size')
    fig.tight_layout()
    fig.savefig("q6_plots.png")
    
    return log_model
"""


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
        """
        # Generate plots for Q6.
        model = train_log_reg_plot(train_exs, dev_exs, feat_extractor, epochs=30)
        """
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model