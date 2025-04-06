import re


string = "@2234324"  # input()
pattern = r"@[0-9a-z]{5,14}"

if re.fullmatch(pattern, string):
    print("Correct")
else:
    print("Incorrect")
