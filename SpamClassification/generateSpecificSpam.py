import random

def generate_spam():
    dict = {
    0:" ",
    1:"x",
    2:"X",
    3:"p",
    4:"P",
    }
    with open("Spam2.txt", "w") as text_file:
        #10000 lines of spam
        for i in range(2000):
            line = ""
            for j in range(random.randint(1,10)):
                line += dict[random.randint(0,len(dict)-1)]
            text_file.write(line +'\n')
generate_spam()
