import sys
import io

with open("unicode_no_num.txt", "w", encoding="utf-8") as text_file:
    for i in range(65536):
        out2 = chr(i)
        if i < 55296 or i> 63744:
            if i != 10:
                text_file.write(out2 +'\n')
text_file.close()
