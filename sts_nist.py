# imports
from nltk import word_tokenize
from nltk.translate.nist_score import sentence_nist
from util import parse_sts
import argparse
import numpy as np


def symmetrical_nist(text_pair):
    """Calculates symmetrical similarity as NIST(a,b) + NIST(b,a)

    Args:
        text_pair (tuple): A tuple of two strings to compare

    Returns:
        float: The symmetrical similarity of the two strings

    Raises:
        ZeroDivisionError: If a division by 0 occurs when computing NIST score
    """

    # save each sentence in the pair to variables
    t1, t2 = text_pair

    # input tokenized text
    t1_toks = word_tokenize(t1.lower())
    t2_toks = word_tokenize(t2.lower())

    # try / except for each side because of ZeroDivisionError
    # 0.0 is lowest score - give that if ZeroDivisionError
    try:
        nist_1 = sentence_nist([t1_toks, ], t2_toks)
    except ZeroDivisionError:
        # print(f'\n\n\nno NIST, {i}')
        nist_1 = 0.0

    try:
        nist_2 = sentence_nist([t2_toks, ], t1_toks)
    except ZeroDivisionError:
        # print(f'\n\n\nno NIST, {i}')
        nist_2 = 0.0

    # return the symmetrical similarity score
    return nist_1 + nist_2


def main(sts_data):
    """Calculates NIST metric for pairs of strings

    Args:
        sts_data (str): The path to the sts benchmark file

    Returns:
        None

    Raises:
        None
    """

    # TODO 1: define a function to read the data in util
    # read the data and returns a list of sentence pairs and similarity scores
    texts, labels = parse_sts(sts_data)
    # output the number of sentence pairs in the data
    print(f"Found {len(texts)} STS pairs\n")

    # take a sample of sentences so the code runs fast for faster debugging
    # when you're done debugging, you may want to run this on more!
    sample_text = texts[120:140]
    sample_labels = labels[120:140]
    # zip them together to make tuples of text associated with labels
    sample_data = zip(sample_labels, sample_text)

    # initialize a list for storing the symmetrical similarity of each pair
    scores = []
    # for the list of labels and list of text pairs in the sample data
    for label, text_pair in sample_data:
        # output the assigned string similarity score
        print(label)
        # output the sentence pair
        print(f"Sentences: {text_pair[0]}\t{text_pair[1]}")
        # TODO 2: Calculate NIST for each pair of sentences
        nist_total = symmetrical_nist(text_pair)
        # compare the assigned similarity score to the symmetrical score
        print(f"Label: {label}, NIST: {nist_total:0.02f}\n")
        # append the symmetrical similarity score for the current text pair
        scores.append(nist_total)

    # grab the first sentence pair
    first_pair = texts[0]
    # output the first sentence pair
    print(first_pair, '\n')
    # save each sentence to variables
    text_a, text_b = first_pair
    # calculate the symmetrical similarity score for each sentence order
    nist_ab = symmetrical_nist((text_a, text_b))
    nist_ba = symmetrical_nist((text_b, text_a))
    # This assertion verifies that symmetrical_nist is symmetrical
    # if the assertion holds, execution continues. If it does not, the program crashes
    assert nist_ab == nist_ba, f"Symmetrical NIST is not symmetrical! Got {nist_ab} and {nist_ba}"

    # TODO 3: find and print the sentences from the sample with the highest and lowest scores
    # find the index of the minimum score
    min_score_index = np.argmin(scores)
    # use the index to extract the minimum score
    min_score = scores[min_score_index]
    # output the lowest score
    print(f'Lowest score: {min_score}')
    # output the sentences from the sample with the lowest score
    print(sample_text[min_score_index], '\n')
    # assert this score = the calculated score of the associated sentence pair
    assert min_score == symmetrical_nist(sample_text[min_score_index])

    # find the index of the maximum score
    max_score_index = np.argmax(scores)
    # use the index to extract the maximum score
    max_score = scores[max_score_index]
    # output the highest score
    print(f'Highest score: {max_score}')
    # output the sentences from the sample with the lowest score
    print(sample_text[max_score_index])
    # assert this score = the calculated score of the associated sentence pair
    assert max_score == symmetrical_nist(sample_text[max_score_index])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="sts data")
    args = parser.parse_args()

    main(args.sts_data)

