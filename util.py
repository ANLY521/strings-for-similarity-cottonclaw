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

    texts = []
    labels = []

    with open(data_file, 'r') as dd:
        for line in dd:
            fields = line.strip().split('\t')
            labels.append(float(fields[4]))
            t1 = fields[5].lower()
            t2 = fields[6].lower()
            texts.append((t1, t2))

    return texts, labels