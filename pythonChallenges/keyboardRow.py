words = ["omk"]

def findWords(words): 
    first_row = {'q','w','e','r','y','u','i','o','p'}
    second_row = {'a','s','d','f','g','h','j','k','l'}
    third_row = {'z','x','c','v','b','n','m'}
    new_words = []

    for word in words:
        if set(word.lower()) - first_row == set(): 
            new_words.append(word)
        elif set(word.lower()) - second_row == set(): 
            new_words.append(word)
        elif set(word.lower()) - third_row == set(): 
            new_words.append(word)
    return new_words

print(findWords(words))

        

