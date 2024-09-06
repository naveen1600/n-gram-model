# n-gram-model

3 Unsmoothed n-grams
To start, you will write a program that computes unsmoothed unigram and bigram probabilities from the training corpus.
You are given a tokenized opinion spam corpus as input (see Section 2). You may want to do additional preprocessing, based on the design decisions you make 1. Make sure to explain preprocessing decisions you make clearly in the report. Note that you may use existing tools just for the purpose of preprocessing but you must write the code for gathering n-gram counts and computing n-gram probabilities yourself.
For example, consider the simple corpus consisting of the sole sentence:
the students like the assignment

Part of what your program would compute for a unigram and bigram model, for example, would be the
following:
P(the) = 0.4, P(like) = 0.2
P(the|like) = 1.0, P(students|the) = 0.5

Preprocessing The files included are already tokenized and hence it should be straightforward to obtain the tokens by using space as the delimiter. Feel free to do any other preprocessing that you might think is important for this corpus. Do not forget to describe and explain your pre-processing choices in your report.

4 Smoothing and unknown words
Firstly, you should implement one or more than one methods to handle unknown words. Then You will need to implement two smoothing methods (e.g. Laplace, Add-k smoothing with different k). Teams can choose any method(s) that they want for each. The report should make clear what methods were selected, providing a description for any non-standard approach (e.g., an approach that was not covered in class or in the class).

5 Perplexity on Validation Set
Implement code to compute the perplexity of a “development set.” (“development set” is just another way to refer to the validation set — part of a dataset that is distinct from the training portion.) Compute and report the perplexity of your model (with variations) on it. Compute perplexity as follows:
P P = (Y
N
i=1
1
P(wi
|wi−1, ..., wi−n+1)
)
1/N
= exp
1
N
X
N
i=1
−log P(wi
|wi−1, ..., wi−n+1)
where N is the total number of tokens in the test corpus and P(wi
|wi−1, ..., wi−n+1) is the n-gram probability of your model. Under the second definition above, perplexity is a function of the average (per-word) log probability: use this to avoid numerical computation errors.
If you experimented with more than one type of smoothing and unknown word handling, you should report and compare the perplexity results of experiments among some of them.
