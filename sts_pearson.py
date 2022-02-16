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
    score_types = ["NIST", "BLEU", "WER", "LCS", "Edit Dist"]

    t1_lows = []
    t2_lows = []
    t1_toks = []
    t2_toks = []
    for text_pair in texts:
        t1, t2 = text_pair
        t1_low = t1.lower()
        t2_low = t2.lower()
        t1_tok = word_tokenize(t1_low)
        t2_tok = word_tokenize(t2_low)
        t1_lows.append(t1_low)
        t2_lows.append(t2_low)
        t1_toks.append(t1_tok)
        t2_toks.append(t2_tok)

    def nist_calc():
        scores = []
        for i in range(len(texts)):
            try:
                nist_ab = sentence_nist([t1_toks[i]], t2_toks[i])
            except ZeroDivisionError:
                nist_ab = 0
            try:
                nist_ba = sentence_nist([t2_toks[i]], t1_toks[i])
            except ZeroDivisionError:
                nist_ba = 0
            # assert nist_ab == nist_ba, f'NIST is not symmetrical! Got {dist_ab} and {dist_ba}'
            scores.append(nist_ab + nist_ba)
        return scores

    def bleu_calc():
        warnings.filterwarnings('ignore')
        scores = []
        sf = SmoothingFunction()
        for i in range(len(texts)):
            try:
                bleu_ab = sentence_bleu([t1_toks[i]], t2_toks[i], smoothing_function=sf.method0)
            except ZeroDivisionError:
                bleu_ab = 0
            try:
                bleu_ba = sentence_bleu([t2_toks[i]], t1_toks[i], smoothing_function=sf.method0)
            except ZeroDivisionError:
                bleu_ba = 0
            # assert bleu_ab == bleu_ba, f'NIST is not symmetrical! Got {dist_ab} and {dist_ba}'
            scores.append(bleu_ab + bleu_ba)
        return scores

    def wer_calc():
        scores = []
        for i in range(len(texts)):
            wer_ab = edit_distance(t1_toks[i], t2_toks[i])/len(t1_toks[i])
            wer_ba = edit_distance(t2_toks[i], t1_toks[i])/len(t2_toks[i])
            # assert len(t1_toks) == len(t2_toks), 'Word Error Rate is not symmetrical! Number of words varies'
            scores.append(wer_ab + wer_ba)
        return scores

    def lcs_calc():
        scores = []
        for i in range(len(texts)):
            LCS_ab = (
                SequenceMatcher(None, t1_lows[i], t2_lows[i])
                .find_longest_match(0, len(t1_lows[i]), 0, len(t2_lows[i]))
                .size
            )
            # LCS_ba = (
            #     SequenceMatcher(None, t2_low, t1_low)
            #     .find_longest_match(0, len(t2_low), 0, len(t1_low))
            #     .size
            # )
            # assert LCS_ab == LCS_ba, f'Longest Common Substring is not symmetrical! Got {LCS_ab} and {LCS_ba}'
            scores.append(LCS_ab)
        return scores

    def edit_dist_calc():
        scores = []
        for i in range(len(texts)):
            dist_ab = edit_distance(t1_lows[i], t2_lows[i])
            # dist_ba = edit_distance(t2_low, t1_low)
            # assert dist_ab == dist_ba, f'Edit Distance is not symmetrical! Got {dist_ab} and {dist_ba}'
            scores.append(dist_ab)
        return scores

    # define a function for calculating each of the metrics for each text pair
    def metric_calc(metric):
        if metric == 'NIST':
            scores = nist_calc()
        elif metric == 'BLEU':
            scores = bleu_calc()
        elif metric == 'WER':
            scores  = wer_calc()
        elif metric == 'LCS':
            scores = lcs_calc()
        else:
            scores = edit_dist_calc()
        return scores

    # TODO 3: Calculate pearson r between each metric and the STS labels and report in the README.
    # Sample code to print results. You can alter the printing as you see fit. It is most important to put the results
    # in a table in the README
    print(f'Semantic textual similarity for {sts_data}\n')
    for metric_name in score_types:
        metric_scores = metric_calc(metric_name)
        score = pearsonr(metric_scores, labels)[0]
        print(f'{metric_name} correlation: {score:.03f}')

    # TODO 4: Complete writeup as specified by TODOs in README (describe metrics; show usage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)

