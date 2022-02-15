from scipy.stats import pearsonr
import argparse
from util import parse_sts


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

    #TODO 3: Calculate pearson r between each metric and the STS labels and report in the README.
    # Sample code to print results. You can alter the printing as you see fit. It is most important to put the results
    # in a table in the README
    print(f"Semantic textual similarity for {sts_data}\n")
    for metric_name in score_types:
        score = 0.0
        print(f"{metric_name} correlation: {score:.03f}")

    # TODO 4: Complete writeup as specified by TODOs in README (describe metrics; show usage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)

