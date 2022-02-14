# TODO: lab, homework
def parse_sts(data_file):
    """Reads a tab-separated sts benchmark file and returns a list of each
       sentence pair and the corresponding similarity score

    Args:
        data_file (str): The path to the sts benchmark file

    Returns:
        tuple[list, list]: A tuple containing a list of sentence pairs and
                           a list of similarity scores

    Raises:
        None
    """

    # initialize a list for storing each sentence-pair tuple
    texts = []
    # initialize a list for storing each sentence-pair similarity score
    labels = []

    # open the benchmark data file for reading
    with open(data_file, 'r') as dd:
        # iterate through each line in the file
        for line in dd:
            # extract each field by stripping whitespace and splitting by tab
            fields = line.strip().split('\t')
            # append the similarity score to the label list
            labels.append(float(fields[4]))
            # extract the first sentence and convert it to lowercase
            t1 = fields[5].lower()
            # extract the second sentence and convert it to lowercase
            t2 = fields[6].lower()
            # place the sentences in a tuple and append it to the texts list
            texts.append((t1, t2))

    # return the tuple containing the list of sentence pairs and labels
    return texts, labels

