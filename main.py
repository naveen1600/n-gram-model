import math
from collections import Counter


class UnigramModel:
    def __init__(self, unigram_freq, total_unigrams):
        self.unigram_freq = unigram_freq
        self.total_unigrams = total_unigrams
        self.add_k = 0
        self.threshold = 0

    def _update_unigram_freq(self):
        # count all the words which occur only once
        self.unigram_freq["<UNK>"] = 0

        keyList = []

        # delete all the keys with value = threshold
        for key in self.unigram_freq:
            if self.unigram_freq[key] <= self.threshold:
                self.unigram_freq["<UNK>"] += 1
                keyList.append(key)

        # delete all the keys that are less or equal to threshold
        for key in keyList:
            del self.unigram_freq[key]

    def handle_unknown_words(self, threshold):
        # those words with frequency less than threshold will be replaced by <UNK> tag

        if threshold < 0:
            raise Exception("Invalid Threshold! It has to be greater than or equal to 0.")

        self.threshold = threshold

        # update the frequency table to include <UNK>
        if self.threshold > 0:
            self._update_unigram_freq()

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
        unigram_count = self.unigram_freq.get(word_list[0], self.unigram_freq['<UNK>'] if self.threshold > 0 else 0)
        V = len(self.unigram_freq.keys())

        # this is using smoothing formula (but if add_k is 0 then it will act as unsmoothed version)
        p = (unigram_count + self.add_k) / (self.total_unigrams + self.add_k * V)

        return p

    def ppl(self, test_unigram_freq, total_tokens):
        log_prob_sum = 0
        # we are doing log calculations to avoid underflow of value
        for unigram in test_unigram_freq:
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

        # return probability
        bigram_count = self.bigram_freq.get(bigram, 0)
        unigram_count = self.unigram_model.unigram_freq.get(word_list[0], 0)
        V = len(self.unigram_model.unigram_freq.keys())

        # this is using smoothing formula (but if add_k is 0 then it will act as unsmoothed version)
        p = (bigram_count + self.add_k) / (unigram_count + self.add_k * V)

        return p

    def ppl(self, test_bigram_freq, total_tokens):
        log_prob_sum = 0
        # we are doing log calculations to avoid underflow of value
        for bigram in test_bigram_freq:
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
test_unigram_freq, test_total_unigrams, test_bigram_freq, test_total_bigrams = CorpusCount(test)

# compute perplexity for unsmoothed

# # TEST 1
#
# print("PPL for unsmoothed Unigram model is {}".format(str(ugm.ppl(test_unigram_freq=test_unigram_freq,
#                                                                   total_tokens=test_total_unigrams))))
# # Exception: Zero probability encountered! Apply unknown word handling
#
# print("PPL for unsmoothed Bigram model is {}".format(str(bgm.ppl(test_bigram_freq=test_bigram_freq,
#                                                                   total_tokens=test_total_unigrams))))
# # Exception: Zero probability encountered! Apply unknown word handling


# TEST 2

# # adding add-1 smoothing
# ugm.set_addk(1)
# bgm.set_addk(1)
#
# print("PPL for (add-1) Smoothed Unigram model is {}".format(str(ugm.ppl(test_unigram_freq=test_unigram_freq,
#                                                                   total_tokens=test_total_unigrams))))
# # PPL for unsmoothed Unigram model is 5.395399727832617
#
# print("PPL for (add-1) Smoothed Bigram model is {}".format(str(bgm.ppl(test_bigram_freq=test_bigram_freq,
#                                                                   total_tokens=test_total_bigrams))))
# # PPL for unsmoothed Bigram model is 164.39628977002462


# # TEST 3
#
# # adding add-k smoothing, k=0.000459
# k = 0.000459
# ugm.set_addk(k)
# bgm.set_addk(k)
#
# print("PPL for (add-1) Smoothed Unigram model is {}".format(str(ugm.ppl(test_unigram_freq=test_unigram_freq,
#                                                                         total_tokens=test_total_unigrams))))
# # PPL for (add-1) Smoothed Unigram model is 6.957609208640156
#
# print("PPL for (add-1) Smoothed Bigram model is {}".format(str(bgm.ppl(test_bigram_freq=test_bigram_freq,
#                                                                        total_tokens=test_total_bigrams))))
# # PPL for (add-1) Smoothed Bigram model is 126.32431475662796

# # TEST 4
#
# # handle unknown words
# ugm.handle_unknown_words(threshold=1)
# bgm.set_addk(k=0.000459)
#
# print("PPL for (add-1) Smoothed Unigram model is {}".format(str(ugm.ppl(test_unigram_freq=test_unigram_freq,
#                                                                         total_tokens=test_total_unigrams))))
# # # PPL for (add-1) Smoothed Unigram model is 3.67328689152463
#
# print("PPL for (add-1) Smoothed Bigram model is {}".format(str(bgm.ppl(test_bigram_freq=test_bigram_freq,
#                                                                        total_tokens=test_total_bigrams))))
# # # PPL for (add-1) Smoothed Bigram model is 117.70706417953151


# TEST 5

# handle unknown words
ugm.handle_unknown_words(threshold=2)
bgm.set_addk(k=0.000459)

print("PPL for (add-1) Smoothed Unigram model is {}".format(str(ugm.ppl(test_unigram_freq=test_unigram_freq,
                                                                        total_tokens=test_total_unigrams))))
# # PPL for (add-1) Smoothed Unigram model is 3.3342323233576603

print("PPL for (add-1) Smoothed Bigram model is {}".format(str(bgm.ppl(test_bigram_freq=test_bigram_freq,
                                                                       total_tokens=test_total_bigrams))))
# # PPL for (add-1) Smoothed Bigram model is 113.09829160338677
