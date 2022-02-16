# imports
from scipy.stats import pearsonr
import argparse
from util import parse_sts
from nltk import word_tokenize
from nltk.translate.nist_score import sentence_nist
import warnings
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.metrics.distance import edit_distance
from difflib import SequenceMatcher


def main(sts_data):
    """Calculates Pearson correlation between semantic similarity scores and
       string similarity metrics

    Args:
        sts_data (str): The path to the STS benchmark partition to implement

    Returns:
        None

    Raises:
        None
    """

    # TODO 1: read the dataset; implement in util.py
    # read the data and return a list of sentence pairs and similarity scores
    texts, labels = parse_sts(sts_data)
    # output the number of sentence pairs in the data set
    print(f"Found {len(texts)} STS pairs\n")

    # TODO 2: Calculate each of the the metrics here for each text pair in the dataset
    # HINT: Longest common substring can be complicated. Investigate difflib.SequenceMatcher for a good option.
    # initialize a list of metrics to calculate
    score_types = ["NIST", "BLEU", "WER", "LCS", "Edit Dist"]

    # initialize a list for all the lower-cased 1st sentences
    t1_lows = []
    # initialize a list for all the lower-cased 2nd sentences
    t2_lows = []
    # initialize a list for all the lower-cased and tokenized 1st sentences
    t1_toks = []
    # initialize a list for all the lower-cased and tokenized 2nd sentences
    t2_toks = []
    # for each text pair in the texts
    for text_pair in texts:
        # extract each sentence
        t1, t2 = text_pair
        # convert the 1st sentence to lowercase
        t1_low = t1.lower()
        # convert the 2nd sentence to lowercase
        t2_low = t2.lower()
        # tokenize the lower-cased 1st sentence
        t1_tok = word_tokenize(t1_low)
        # tokenize the lower-cased 2nd sentence
        t2_tok = word_tokenize(t2_low)
        # append the lower-cased 1st sentence to the correct list
        t1_lows.append(t1_low)
        # append the lower-cased 2nd sentence to the correct list
        t2_lows.append(t2_low)
        # append the lower-cased and tokenized 1st sentence to the correct list
        t1_toks.append(t1_tok)
        # append the lower-cased and tokenized 2nd sentence to the correct list
        t2_toks.append(t2_tok)

    # define a function for calculating NIST
    def nist_calc():
        # initialize a list to store NIST scores
        scores = []
        # for each index in the list of texts
        for i in range(len(texts)):
            # try to determine NIST score for the ith tokenized sentence pair
            try:
                nist_ab = sentence_nist([t1_toks[i]], t2_toks[i])
            # if this results in a division by 0, just assign the score as 0
            except ZeroDivisionError:
                nist_ab = 0
            # repeat with sentence order reversed since NIST is not symmetrical
            try:
                nist_ba = sentence_nist([t2_toks[i]], t1_toks[i])
            except ZeroDivisionError:
                nist_ba = 0
            # assert nist_ab == nist_ba, f'NIST is not symmetrical! Got {dist_ab} and {dist_ba}'
            # calc the score as the sum of the scores for each sentence order
            scores.append(nist_ab + nist_ba)
        # return the list of NIST scores for each text pair in the data set
        return scores

    # define a function for calculating BLEU
    def bleu_calc():
        # ignore warnings about orders of n-gram with no overlap
        warnings.filterwarnings('ignore')
        # initialize a list to store BLEU scores
        scores = []
        # use a function to prevent a score of 0 if there is no n-gram overlap
        sf = SmoothingFunction()
        # for each index in the list of texts
        for i in range(len(texts)):
            # try to determine BLEU score for the ith tokenized sentence pair
            try:
                bleu_ab = sentence_bleu([t1_toks[i]], t2_toks[i], smoothing_function=sf.method0)
            # if this results in a division by 0, just assign the score as 0
            except ZeroDivisionError:
                bleu_ab = 0
            # repeat with sentence order reversed since BLEU is not symmetrical
            try:
                bleu_ba = sentence_bleu([t2_toks[i]], t1_toks[i], smoothing_function=sf.method0)
            except ZeroDivisionError:
                bleu_ba = 0
            # assert bleu_ab == bleu_ba, f'NIST is not symmetrical! Got {dist_ab} and {dist_ba}'
            # calc the score as the sum of the scores for each sentence order
            scores.append(bleu_ab + bleu_ba)
        # return the list of BLEU scores for each text pair in the data set
        return scores

    # define a function for calculating WER
    def wer_calc():
        # initialize a list to store WER scores
        scores = []
        # for each index in the list of texts
        for i in range(len(texts)):
            # calc WER as word-level edit distance / # words in reference
            wer_ab = edit_distance(t1_toks[i], t2_toks[i])/len(t1_toks[i])
            # repeat with sentence order reversed since WER is not symmetrical
            wer_ba = edit_distance(t2_toks[i], t1_toks[i])/len(t2_toks[i])
            # assert len(t1_toks) == len(t2_toks), 'Word Error Rate is not symmetrical! Number of words varies'
            # calc the score as the sum of the scores for each sentence order
            scores.append(wer_ab + wer_ba)
        # return the list of WER scores for each text pair in the data set
        return scores

    # define a function for calculating LCS
    def lcs_calc():
        # initialize a list to store LCS scores
        scores = []
        # for each index in the list of texts
        for i in range(len(texts)):
            # calculate LCS by:
            LCS_ab = (
                # making a SequenceMatcher object w/ the lower-cased sentences
                SequenceMatcher(None, t1_lows[i], t2_lows[i])
                # using the object to find their longest common substring
                .find_longest_match(0, len(t1_lows[i]), 0, len(t2_lows[i]))
                # determining the size of the longest common substring
                .size
            )
            # LCS_ba = (
            #     SequenceMatcher(None, t2_low, t1_low)
            #     .find_longest_match(0, len(t2_low), 0, len(t1_low))
            #     .size
            # )
            # assert LCS_ab == LCS_ba, f'Longest Common Substring is not symmetrical! Got {LCS_ab} and {LCS_ba}'
            # calc score as LCS of original sentence order (LCS is symmetrical)
            scores.append(LCS_ab)
        # return the list of LCS scores for each text pair in the data set
        return scores

    # define a function for calculating Levenshtein Edit Distance (LED)
    def edit_dist_calc():
        # initialize a list to store edit dist scores
        scores = []
        # for each index in the list of texts
        for i in range(len(texts)):
            # calculate sentence-level edit dist
            dist_ab = edit_distance(t1_lows[i], t2_lows[i])
            # dist_ba = edit_distance(t2_low, t1_low)
            # assert dist_ab == dist_ba, f'LED is not symmetrical! Got {dist_ab} and {dist_ba}'
            # calc score as LED of original sentence order (LED is symmetrical)
            scores.append(dist_ab)
        # return the list of LED scores for each text pair in the data set
        return scores

    # define a function for calculating each of the metrics for each text pair
    def metric_calc(metric):
        # if the metric is NIST:
        if metric == 'NIST':
            # call "nist_calc" to return the NIST scores for each text pair
            scores = nist_calc()
        # if the metric is BLEU:
        elif metric == 'BLEU':
            # call "bleu_calc" to return the BLEU scores for each text pair
            scores = bleu_calc()
        # if the metric is WER:
        elif metric == 'WER':
            # call "wer_calc" to return the WER scores for each text pair
            scores  = wer_calc()
        # if the metric is LCS:
        elif metric == 'LCS':
            # call "lcs_calc" to return the LCS scores for each text pair
            scores = lcs_calc()
        # if the metric is not one of the four above, then it is LED
        else:
            # call "edit_dist_calc" to return the LED scores for each text pair
            scores = edit_dist_calc()
        # return the scores for each text pair based on the metric called
        return scores

    # TODO 3: Calculate pearson r between each metric and the STS labels and report in the README.
    # Sample code to print results. You can alter the printing as you see fit. It is most important to put the results
    # in a table in the README
    # output the partition of the data set that is being used to calculate STS
    print(f'Semantic textual similarity for {sts_data}\n')
    # for each metric in the list of metrics:
    for metric_name in score_types:
        # call "metric_calc" to determine scores based on the current metric
        metric_scores = metric_calc(metric_name)
        # calculate Pearson r between the current metric and the STS labels
        score = pearsonr(metric_scores, labels)[0]
        # output Pearson r between the current metric and the STS labels
        print(f'{metric_name} correlation: {score:.03f}')

    # TODO 4: Complete writeup as specified by TODOs in README (describe metrics; show usage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)

