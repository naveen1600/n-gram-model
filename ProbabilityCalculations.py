from collections import Counter

unigrams = {}
bigrams = dict()

def CorpusCount(strs : list[list[str]]):  # strs contains list of word in a sentence
    global unigrams
    global bigrams

    wrapped_lists = []
    
    for list in strs:
        wrapped_sublist = ['<s>'] + list + ['</s>']
        wrapped_lists.append(wrapped_sublist)         # adding <s> and </s> to each sentence


    flatten_lists = [word for list in wrapped_lists for word in list]         # converting it to a single array
    
    unigrams = Counter(flatten_lists)        # dictionary to store the unigrams and their corresponding frequency of each word
        
    

    previousWord = None

    for currentWord in flatten_lists:
        if previousWord is not None and currentWord != '<s>':         
            bigram = previousWord + ' ' + currentWord             # adding all possible two adjacent word combinations to the bigrams dictionary
            if bigram in bigrams:
                bigrams[bigram] += 1
            else:
                bigrams[bigram] = 1
        previousWord = currentWord
    
  
def probCalcUnsmootened(strs : str) -> int:
    probability = bigrams[strs]/unigrams[strs.split(' ')[0]]    # probability calculation for unsmootened bigram model
    return probability



s1 = "i love utd"
s2 = "i go to utd"
s3 = "i love apples"
list1 = s1.split(' ')
list2 = s2.split(' ')
list3 = s3.split(' ')
lists = [list1, list2, list3]
CorpusCount(lists)
print(probCalcUnsmootened("i love"))
