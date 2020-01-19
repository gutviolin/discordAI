import random

def generate_spam():
    dict = {
    0:" ",
    1:"x",
    2:"X",
    3:"p",
    4:"P",
    }
    with open("NotSpam3.txt", "w") as text_file:
        #10000 lines of spam
        for i in range(10000):
            line = "!levels"
            text_file.write(line +'\n')

generate_spam()
