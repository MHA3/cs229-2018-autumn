import collections

import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    return message.lower().split()
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message. 

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    from collections import defaultdict
    word_dict = defaultdict(int)
    for m in messages:
        for w in get_words(m):
            word_dict[w] += 1

    index = 0
    for k in list(word_dict.keys()):
        if word_dict[k] >= 5:
            word_dict[k] = index
            index += 1
        else:
            del word_dict[k]

    return word_dict
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each 
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that 
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
    """
    # *** START CODE HERE ***
    X = np.zeros((len(messages), len(word_dictionary)))

    for i, m in enumerate(messages):
        for w in get_words(m):
            if w in word_dictionary:
                X[i][word_dictionary[w]] += 1

    return X
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***

    # n, V = matrix.shape

    # matrix_y1 = matrix[labels==1, :].sum(axis=0)
    # matrix_y0 = matrix[labels==0, :].sum(axis=0)

    # phi_k_y1 = (matrix_y1 + 1) / (matrix_y1.sum() + V)
    # phi_k_y0 = (matrix_y0 + 1) / (matrix_y0.sum() + V)

    # phi_y = np.mean(labels)

    # print(f"(phi_k_y1, phi_k_y0, phi_y) = {(phi_k_y1, phi_k_y0, phi_y)}")

    # ==================================================

    # print(f"naive bayes train matrix shape: {matrix.shape}")
    prior_prob_spam = sum(labels)/len(labels)
    prior_prob_non_spam = 1 - prior_prob_spam

    # print(f"spam matrix shape: {matrix[labels==1].shape}")
    num_words_in_spam_docs = np.sum(matrix[labels==1], axis=0)
    # print(f"num_words_in_spam_docs shape = {num_words_in_spam_docs.shape}")
    total_words_spam = sum(num_words_in_spam_docs)
    prob_word_given_spam = (num_words_in_spam_docs + 1)/(total_words_spam + len(num_words_in_spam_docs))

    num_words_in_non_spam = np.sum(matrix[labels==0], axis=0)
    total_words_non_spam = sum(num_words_in_non_spam)
    # print(f"num_words_in_non_spam shape = {num_words_in_non_spam.shape}")
    prob_word_given_non_spam = (num_words_in_non_spam + 1)/(total_words_non_spam + len(num_words_in_non_spam))
    
    # print(f"(prob_word_given_spam, prob_word_given_non_spam, prior_prob_spam) = {(prob_word_given_spam, prob_word_given_non_spam, prior_prob_spam)}")
    # we will use logs later, so lets apply logs here
    return {
                "prior_log_prob_spam": np.log(prior_prob_spam), 
                "prior_log_prob_non_spam": np.log(prior_prob_non_spam), 
                "log_prob_word_given_spam": np.log(prob_word_given_spam), 
                "log_prob_word_given_non_spam": np.log(prob_word_given_non_spam)
            }
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    log_prob_spam = model["prior_log_prob_spam"] + np.sum(matrix*model["log_prob_word_given_spam"], axis=1)
    log_prob_non_spam = model["prior_log_prob_non_spam"] + np.sum(matrix*model["log_prob_word_given_non_spam"], axis=1)

    preds = log_prob_spam >= log_prob_non_spam

    # testing the assumption of mutual exclusivity and exhaustivity
    # the following assertion is invalid because these aren't actual probs, because
    # that would also require considering the total number of words. The way the model
    # is implemented currently, the longer the message, the lower the log prob for both
    # spam and non_spam classes.
    # test_preds = log_prob_spam >= np.log(0.5)
    # assert all(preds == test_preds), f"The preds via log_prob_spam >= log_prob_non_spam should be the same as that from log_prob_spam >= log(0.5)"

    return preds
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in 6c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: The top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    log_indication_frac = model["log_prob_word_given_spam"] - model["log_prob_word_given_non_spam"]
    top_5_idx = np.argsort(log_indication_frac)[-5:]
    # print(top_5_idx)
    dict_idx_to_word = {v: k for k, v in dictionary.items()}
    top_5_words = [dict_idx_to_word[i] for i in top_5_idx]
    return top_5_words[::-1]
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider
    
    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    best_rad = float('inf')
    best_acc = 0
    for r in radius_to_consider:
        val_preds = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, r)
        acc = np.mean(val_preds == val_labels)
        if acc > best_acc:
            best_acc = acc
            best_rad = r
    return best_rad
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('../data/ds6_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('../data/ds6_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('../data/ds6_test.tsv')
    
    dictionary = create_dictionary(train_messages)

    util.write_json('./output/p06_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('./output/p06_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('./output/p06_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('./output/p06_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('./output/p06_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
