def get_letter_dict():
    with open('unicode_no_num.txt', 'r', encoding="utf-8") as file:
        array_of_content = [line.splitlines() for line in file]
        rows = len(array_of_content)
        char_array = []
        for i in range(rows):
            ascii_char = array_of_content[i][0]
            char_array.append(ascii_char)
    dictOfWords = { char_array[k] : k for k in range(0, len(char_array) ) }
    #dictOfWords = { i : array_of_content[i] for i in range(0, len(array_of_content) ) }
    #print (dictOfWords)
    return dictOfWords

def get_letter_dict_with_reserved():
    letter_index = get_letter_dict()
    letter_index = {k:(v+4) for k,v in letter_index.items()}
    letter_index["<PAD>"] = 0
    letter_index["<START>"] = 1
    letter_index["<UNK>"] = 2  # unknown
    letter_index["<UNUSED>"] = 3
    return letter_index

#print(get_letter_dict())
print(get_letter_dict_with_reserved())
