from scipy.stats import pearsonr
import argparse
from util import parse_sts
from nltk import word_tokenize
from nltk.translate.nist_score import sentence_nist
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


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
    score_types = ["NIST", "BLEU", "Word Error Rate", "Longest common substring", "Edit Distance"]

    # NIST
    """
    scores = []
    for text_pair in texts:
        t1, t2 = text_pair
        t1_toks = word_tokenize(t1.lower())
        t2_toks = word_tokenize(t2.lower())
        try:
            nist_ab = sentence_nist([t1_toks], t2_toks)
        except ZeroDivisionError:
            nist_ab = 0
        try:
            nist_ba = sentence_nist([t2_toks], t1_toks)
        except ZeroDivisionError:
            nist_ba = 0
        # assert nist_ab == nist_ba, f'Symmetrical NIST is not symmetrical! Got {nist_ab} and {nist_ba}'
        scores.append(nist_ab)
    score = pearsonr(scores, labels)[0]
    print(f'NIST correlation: {score:.03f}')
    """

    # BLEU
    scores = []
    sf = SmoothingFunction()
    for text_pair in texts:
        t1, t2 = text_pair
        t1_toks = word_tokenize(t1.lower())
        t2_toks = word_tokenize(t2.lower())
        try:
            bleu_ab = sentence_bleu([t1_toks], t2_toks, smoothing_function=sf.method0)
        except ZeroDivisionError:
            bleu_ab = 0
        try:
            bleu_ba = sentence_bleu([t2_toks], t1_toks, smoothing_function=sf.method0)
        except ZeroDivisionError:
            bleu_ba = 0
        # assert bleu_ab == bleu_ba, f'Symmetrical NIST is not symmetrical! Got {nist_ab} and {nist_ba}'
        scores.append(bleu_ab)
    score = pearsonr(scores, labels)[0]
    print(f'BLEU correlation: {score:.03f}')

    # WER
    # LCS
    # ED

    # define a function for calculating either NIST or BLEU metric
    def nist_or_bleu_calc(text_pair, metric):
        print('Function is not ready yet')

    # define a function for calculating each of the metrics for each text pair
    def metric_calc(metric):
        if metric == 'NIST' or metric == 'BLEU':
            scores = [nist_or_bleu_calc(text_pair, metric) for text_pair in texts]



    # TODO 3: Calculate pearson r between each metric and the STS labels and report in the README.
    # Sample code to print results. You can alter the printing as you see fit. It is most important to put the results
    # in a table in the README
    print(f"Semantic textual similarity for {sts_data}\n")
    for metric_name in score_types:
        metric_scores = metric_calc(metric_name)
        score = pearsonr(metric_scores, labels)
        print(f"{metric_name} correlation: {score:.03f}")

    # TODO 4: Complete writeup as specified by TODOs in README (describe metrics; show usage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)

