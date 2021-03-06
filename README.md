Semantic textual similarity using string similarity
---------------------------------------------------

This project examines string similarity metrics for semantic textual similarity.
Though semantics go beyond the surface representations seen in strings, some of these
metrics constitute a good benchmark system for detecting STS.

Data is from the [STS benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark).
<br><br><br>


## Metrics

Below is a brief description of the metrics used to evaluate
string similarity between sentences contained in various
partitions of the STS benchmark data set.
<br><br>
<h3>NIST</h3>
NIST is a formula used for Machine Translation (MT) evaluation.
This metric was developed by the National Institute of
Standards and Technology (NIST), but it actually derives from
an MT evaluation metric, called Bilingual Evaluation
Understudy (BLEU), that was developed by the International
Business Machines Corporation (IBM). Although describing the
formula in words is beyond the scope of this description, the
computation for NIST is largely dependent on the percentage of
N-grams in a translation string that also occur in the
corresponding reference string.
<br><br>
<h3>BLEU</h3>
As mentioned above, BLEU is an MT evaluation metric developed
by IBM that predates the formula that NIST uses. Like NIST,
BLEU takes into consideration the fraction of translation
N-grams that match the set of N-grams in the corresponding
reference text. However, unlike NIST, BLEU gives equal
weighting to N-grams, whereas the weighting in NIST is partial
to N-grams that have a lower frequency in the data set.
<br><br>
<h3>Edit Distance</h3>
Edit Distance, also known as Levenshtein Distance, is a string
similarity metric that was developed by Vladimir Levenshtein
in 1966. This metric quantifies the similarity between two
strings by assessing the number of character-level deletions,
insertions, and substitutions that would be required to
transform one of the strings into the other. Unlike NIST and
BLEU, Edit Distance is symmetrical, so the quantity for this
metric is independent of the string that is chosen to be
transformed.
<br><br>
<h3>WER</h3>
Word Error Rate (WER) is another technique for calculating the
similarity between two strings. This method works much like
edit distance, but rather than evaluating similarity at the
<i>character</i>-level, the computation for WER determines the
number of <i>word</i>-level deletions, insertions, and
substitutions that would be required to transform a reference
string into a hypothesis string. To fully compute WER, this
word-level edit distance is then divided by the number of
words in the reference string. Since the number of words
between two strings often varies, WER is not symmetrical;
therefore, unlike Edit Distance, WER depends on the string
that is chosen to be transformed.
<br><br>
<h3>LCS</h3>
Longest Common Substring (LCS) is yet another strategy for
judging the similarity between a set of strings. As the name
implies, LCS is calculated by identifying the length of the
longest consecutive character string that is shared between a
reference text and a hypothesis text. Like Edit Distance, LCS
is symmetrical, so the value for this approach to configuring
string similarity is irrelevant of the text that is assigned
as the reference.
<br><br>
<h3>Sources for Descriptions</h3>
<ul>
<li>Doddington, George. "Automatic Evaluation of Machine
Translation Quality Using N-Gram Co-Occurrence Statistics."
<i>Proceedings of the Second International Conference on Human
Language Technology Research,</i> 2002, pp. 138-145,
https://dl.acm.org/doi/10.5555/1289189.1289273. Accessed 16
Feb. 2022.</li>
<li>Jurafsky and Martin, Chapter 6</li>
<li>Lecture Slides - Module 3, Week 1</li>
</ul>
<br><br>



## Correlations

| Metric    | Train | Dev | Test |
|-----------|-------| --- | ---- |
| NIST      | 0.496 | 0.593 | 0.475 |
| BLEU      | 0.371 | 0.433 | 0.353 |
| WER       | -0.353 | -0.452 | -0.358 |
| LCS       | 0.362 | 0.468 | 0.347 |
| Edit Dist | 0.033 | -0.175 | -0.039 |

<br><br>


## Example Usage

<ul>
<li>Training Set Correlations: <code>python sts_pearson.py --sts_data stsbenchmark/sts-train.csv</code></li>
<li>Development Set Correlations: <code>python sts_pearson.py --sts_data stsbenchmark/sts-dev.csv</code></li>
<li>Testing Set Correlations: <code>python sts_pearson.py --sts_data stsbenchmark/sts-test.csv</code></li>
</ul>
<br><br>



## lab, week 1: sts_nist.py

Calculates NIST machine translation metric for sentence pairs in an STS dataset.

Example usage:

`python sts_nist.py --sts_data stsbenchmark/sts-dev.csv`



## lab, week 2: sts_tfidf.py

Calculate pearson's correlation of semantic similarity with TFIDF vectors for text.



## homework, week 1: sts_pearson.py

Calculate pearson's correlation of semantic similarity with the metrics specified in the starter code.
Calculate the metrics between lowercased inputs and ensure that the metric is the same for either order of the 
sentences (i.e. sim(A,B) == sim(B,A)). If not, use the strategy from the lab.
Use SmoothingFunction method0 for BLEU, as described in the nltk documentation.

Run this code on the three partitions of STSBenchmark to fill in the correlations table above.
Use the --sts_data flag and edit PyCharm run configurations to run against different inputs,
 instead of altering your code for each file.