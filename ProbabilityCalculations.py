from collections import Counter


def probabilityCalc(strs : list[list[str]]):  # strs contains list of word in a sentence
    wrapped_lists = []
    
    for list in strs:
        wrapped_sublist = ['<s>'] + list + ['</s>']
        wrapped_lists.append(wrapped_sublist)         # adding <s> and </s> to each sentence


    flatten_lists = [word for list in wrapped_lists for word in list]         # converting it to a single array
    
    unigrams = Counter(flatten_lists)        # dictionary to store the unigrams and their corresponding frequency of each word
        
    print(unigrams)


    i = 0
    j = 1

    

    


s1 = "the quick brown fox jumped over the lazy dog"
s2 = "waltz bad nymph for quick jigs vex"
list1 = s1.split(' ')
list2 = s2.split(' ')
lists = [list1, list2]
probabilityCalc(lists)