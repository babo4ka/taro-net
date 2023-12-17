import os
import re

descs_path = "../cards_descs"

for filename in os.listdir(descs_path):
    with open(("../cards_descs/" + filename), encoding='utf-8') as f:
        lines = [line.rstrip() for line in f]
        lines.pop(0)
        for l in lines:
            if l == "===":
                lines.remove(l)

        for l in lines:
            print(re.split('[^a-zа-яё]+', l, flags=re.IGNORECASE))

    print("=============================================================")
