
# N-Gram Model

## 1. Unsmoothed n-grams
A program that computes unsmoothed unigram and bigram probabilities from the training corpus. 
We are given a tokenized opinion spam corpus as input.

For example, consider the simple corpus consisting of the sole sentence: 
*the students like the assignment*

Part of what our program would compute probability for a unigram and bigram model would be the following:

- P(the) = 0.4
- P(like) = 0.2
- P(the | like) = 1.0
- P(students | the) = 0.5

## 2. Preprocessing
The files included are already tokenized, so it should be straightforward to obtain the tokens by using space as the delimiter.

## 3. Smoothing and Unknown Words
Firstly, we have implement one or more methods to handle unknown words. Then, implement two smoothing methods (e.g. Laplace, Add-k smoothing with different k).

## 4. Perplexity on Validation Set
We compute the perplexity of a “development set” (another term for validation set — part of a dataset distinct from the training portion).

Perplexity as follows:

$$
P = \exp\left(\frac{1}{N} \sum_{i=1}^{N} -\log P(w_i \mid w_{i-1}, \dots, w_{i-n+1})\right)
$$



Where N is the total number of tokens in the test corpus, and P(wi | wi-1, ..., wi-n+1) is the n-gram probability of your model. 
Under this definition, perplexity is a function of the average (per-word) log probability, which helps avoid numerical computation errors.
