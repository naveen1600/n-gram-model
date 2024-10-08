import math
from collections import Counter


class UnigramModel:
    def __init__(self, unigram_freq, total_unigrams):
        self.unigram_freq = unigram_freq
        self.total_unigrams = total_unigrams
        self.add_k = 0

    def set_addk(self, k):
        if 0 <= k <= 1:
            self.add_k = k
        else:
            raise Exception('Invalid k value. The value of k should be between 0 and 1.')

    def probability(self, unigram):
        # check if ngram is a single word or two word
        word_list = unigram.split(" ")
        if len(word_list) != 1:
            raise Exception("Unigram probability needs one word exactly.")

        # return probability
        unigram_count = self.unigram_freq.get(word_list[0], self.unigram_freq['<UNK>'])
        V = len(self.unigram_freq.keys())

        # this is using smoothing formula (but if add_k is 0 then it will act as unsmoothed version)
        p = (unigram_count + self.add_k) / (self.total_unigrams + self.add_k * V)

        return p

    def ppl(self, test_tokens, total_tokens):
        log_prob_sum = 0
        # we are doing log calculations to avoid underflow of value
        for unigram in test_tokens:
            p = self.probability(unigram)
            if p > 0:
                log_prob_sum += math.log(p)
            else:
                raise Exception("Zero probability encountered! Apply unknown word handling")

        return math.exp(-1 * log_prob_sum / total_tokens)


class BigramModel:
    def __init__(self, bigram_freq, total_bigrams, unigram_model):
        self.unigram_model = unigram_model
        self.bigram_freq = bigram_freq
        self.total_bigrams = total_bigrams
        self.add_k = 0

    def set_addk(self, k):
        if 0 <= k <= 1:
            self.add_k = k
        else:
            raise Exception('Invalid k value. The value of k should be between 0 and 1.')

    def probability(self, bigram):
        # check if ngram is a single word or two word
        word_list = bigram.split(" ")
        if len(word_list) != 2:
            raise Exception("Bigram probability needs two words exactly.")

        # handle unknown
        if word_list[0] not in self.unigram_model.unigram_freq:
            word_list[0] = "<UNK>"
        if word_list[1] not in self.unigram_model.unigram_freq:
            word_list[1] = "<UNK>"

        bigram = " ".join(word_list)

        # return probability
        bigram_count = self.bigram_freq.get(bigram, 0)
        unigram_count = self.unigram_model.unigram_freq.get(word_list[0], 0)
        V = len(self.unigram_model.unigram_freq.keys())

        # this is using smoothing formula (but if add_k is 0 then it will act as unsmoothed version)
        p = (bigram_count + self.add_k) / (unigram_count + self.add_k * V)

        return p

    def ppl(self, test_tokens, total_tokens):
        log_prob_sum = 0
        # we are doing log calculations to avoid underflow of value
        for bigram in test_tokens:
            p = self.probability(bigram)
            if p > 0:
                log_prob_sum += math.log(p)
            else:
                raise Exception("Zero probability encountered! Apply unknown word handling")

        return math.exp(-1 * log_prob_sum / total_tokens)


def processData(filePath):
    data = []

    with open(filePath, "r") as file:
        content = file.readlines()

    for line in content:
        line = line.lower()
        data.append(["<s>"] + line.strip("\n").split(" ") + ["</s>"])

    flattened_data = [word for review in data for word in review]

    return flattened_data


def CorpusCount(data):
    unigrams = Counter(data)
    total_unigrams = len(data)
    bigrams = {}
    total_bigrams = 0
    previousWord = None
    for currentWord in data:
        if previousWord is not None and currentWord != '<s>':
            bigram = previousWord + ' ' + currentWord
            total_bigrams += 1
            if bigram in bigrams:
                bigrams[bigram] += 1
            else:
                bigrams[bigram] = 1
        previousWord = currentWord
    return unigrams, total_unigrams, bigrams, total_bigrams


def generate_bigrams(data):
    bigrams = []
    total_bigrams = 0
    previousWord = None
    for currentWord in data:
        if previousWord is not None and currentWord != '<s>':
            bigram = previousWord + ' ' + currentWord
            total_bigrams += 1
            bigrams.append(bigram)
        previousWord = currentWord
    return bigrams, total_bigrams


def handle_unknown_words(unigram_freq, threshold, tokenList):
    for index, token in enumerate(tokenList):
        if unigram_freq[token] <= threshold:
            tokenList[index] = "<UNK>"
    return tokenList


# load training data
dataset_path = "dataset/train.txt"
train = processData(dataset_path)

# get the count of each unigram and bigram
unigram_freq, total_unigrams, bigram_freq, total_bigrams = CorpusCount(train)

# unsmoothed unigram and bigram model
ugm = UnigramModel(unigram_freq=unigram_freq, total_unigrams=total_unigrams)
bgm = BigramModel(bigram_freq=bigram_freq, total_bigrams=total_bigrams, unigram_model=ugm)

# Testing the model

# Load test data
test_path = "dataset/val.txt"
test = processData(test_path)

# generate test unigram and bigram freq
test_bigrams, test_total_bigrams = generate_bigrams(test)
test_unigrams, test_total_unigrams = test, len(test)

# compute perplexity for unsmoothed
#
# # TEST 1
#
# print("PPL for unsmoothed Unigram model is {}".format(str(ugm.ppl(test_tokens=test_unigrams,
#                                                                   total_tokens=test_total_unigrams))))
# # Exception: Zero probability encountered! Apply unknown word handling
#
# print("PPL for unsmoothed Bigram model is {}".format(str(bgm.ppl(test_tokens=test_bigrams,
#                                                                   total_tokens=test_total_bigrams))))
# # Exception: Zero probability encountered! Apply unknown word handling


# # TEST 2
#
# # adding add-1 smoothing
# ugm.set_addk(1)
# bgm.set_addk(1)
#
# print("PPL for (add-1) Smoothed Unigram model is {}".format(str(ugm.ppl(test_tokens=test_unigrams,
#                                                                   total_tokens=test_total_unigrams))))
# # PPL for unsmoothed Unigram model is 461.08291192772447
#
# print("PPL for (add-1) Smoothed Bigram model is {}".format(str(bgm.ppl(test_tokens=test_bigrams,
#                                                                   total_tokens=test_total_bigrams))))
# # PPL for unsmoothed Bigram model is 1048.1314967506887


# # TEST 3
#
# # adding add-k smoothing, k=0.000459
# k = 0.000459
# ugm.set_addk(k)
# bgm.set_addk(k)
#
# print("PPL for (add-k) Smoothed Unigram model is {}".format(str(ugm.ppl(test_tokens=test_unigrams,
#                                                                         total_tokens=test_total_unigrams))))
# # PPL for (add-k) Smoothed Unigram model is 574.7083806922085
#
# print("PPL for (add-k) Smoothed Bigram model is {}".format(str(bgm.ppl(test_tokens=test_bigrams,
#                                                                        total_tokens=test_total_bigrams))))
# # PPL for (add-k) Smoothed Bigram model is 376.25218792927654

# # TEST 4

# # handle unknown words
# unk_train = handle_unknown_words(unigram_freq=unigram_freq, threshold=1, tokenList=train)
# unigram_freq_unk, total_unigrams_unk, bigram_freq_unk, total_bigrams_unk = CorpusCount(unk_train)
# ugm2 = UnigramModel(unigram_freq=unigram_freq_unk, total_unigrams=total_unigrams_unk)
# bgm2 = BigramModel(bigram_freq=bigram_freq_unk, total_bigrams=total_bigrams_unk, unigram_model=ugm2)
#
# print("PPL for unknown word handled Unigram model is {}".format(str(ugm2.ppl(test_tokens=test_unigrams,
#                                                                          total_tokens=test_total_unigrams))))
# # PPL for unknown word handled Unigram model is 292.1907013179355
#
# print("PPL for unknown word handled Bigram model is {}".format(str(bgm2.ppl(test_tokens=test_bigrams,
#                                                                         total_tokens=test_total_bigrams))))
# # Exception: Zero probability encountered! Apply unknown word handling


# TEST 5

# handle unknown words
unk_train = handle_unknown_words(unigram_freq=unigram_freq, threshold=1, tokenList=train)
unigram_freq_unk, total_unigrams_unk, bigram_freq_unk, total_bigrams_unk = CorpusCount(unk_train)
ugm2 = UnigramModel(unigram_freq=unigram_freq_unk, total_unigrams=total_unigrams_unk)
bgm2 = BigramModel(bigram_freq=bigram_freq_unk, total_bigrams=total_bigrams_unk, unigram_model=ugm2)
bgm2.set_addk(k=0.000459)

print("PPL for (add-k) Smoothed Unigram model is {}".format(str(ugm2.ppl(test_tokens=test_unigrams,
                                                                         total_tokens=test_total_unigrams))))
# PPL for (add-k) Smoothed Unigram model is 292.1907013179355

print("PPL for (add-k) Smoothed Bigram model is {}".format(str(bgm2.ppl(test_tokens=test_bigrams,
                                                                        total_tokens=test_total_bigrams))))
# PPL for (add-k) Smoothed Bigram model is 201.31185045640476

