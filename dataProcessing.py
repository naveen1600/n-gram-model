"""
Processing Decisions:
1. We will consider each token as a word that is separated by space. (Even the !.,"" ...)
2. We will put "<s>" "</s>" at the start of a review and the end of a review respectively.
"""


def processData(filePath):
    dataset = []

    with open(filePath, "r") as file:
        content = file.readlines()
        for line in content:
            dataset.append(["<s>"] + line.strip("\n").split(" ") + ["</s>"])

    return dataset

