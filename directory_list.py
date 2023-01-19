# import os
# # path = 'E:/Assignment_/SEM_6/Minor_2/Foods_DS/'
# #
# # for root, directories, files in os.walk(path, topdown=False):
# #     for name in files:
# #         print(os.path.join(root, name))
# #     for name in directories:
# #         print(os.path.join(root, name))
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
# import tensorflow as tf
# tf.test.is_gpu_available


import requests
from bs4 import BeautifulSoup

# enter city name
city = "lucknow"

# creating url and requests instance
url = "https://www.google.com/search?q="+"weather"+city
html = requests.get(url).content

# getting raw data
soup = BeautifulSoup(html, 'html.parser')
temp = soup.find('div', attrs={'class': 'BNeawe iBp4i AP7Wnd'}).text
str = soup.find('div', attrs={'class': 'BNeawe tAd8D AP7Wnd'}).text

# formatting data
data = str.split('\n')
time = data[0]
sky = data[1]

# getting all div tag
listdiv = soup.findAll('div', attrs={'class': 'BNeawe s3v9rd AP7Wnd'})
strd = listdiv[5].text

# getting other required data
pos = strd.find('Wind')
other_data = strd[pos:]

# printing all data
print("Temperature is", temp)
print("Time: ", time)
print("Sky Description: ", sky)
print(other_data)

