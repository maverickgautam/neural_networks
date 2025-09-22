import os
import urllib.request
import ssl
from enum import verify
#Download : https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/the-verdict.txt

filepath = "/Users/kunalgau/Downloads/the-verdict.txt"
url = "https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists(filepath):
    with urllib.request.urlopen(url) as response :
        text_data = response.read().decode("utf-8")
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(text_data)
else:
    with open(filepath, "r", encoding="utf-8") as file:
        text_data = file.read()
        print(text_data)
        print(text_data[-99:])
