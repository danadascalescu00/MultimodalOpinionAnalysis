from emoticons_emojis import *

import os
import sys
import glob, shutil
import pickle
import argparse
from copy import deepcopy

import re
import string
import json
import unicodedata

import cv2 as cv

import random
import numpy as np
import pandas as pd
from matplotlib import pyplot

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

base_dir = 'C:/Users/Dana/Desktop/Licenta/'
additional_files_dir = os.path.join(base_dir, 'OpinionMining/AdditionalFiles')
datasets_names = ['MVSA_Single', 'MVSA_Multi']

punctuation = string.punctuation
sentiment_label = {"negative": 0, "positive": 1, "neutral": 2}

STOPWORDS = set(stopwords.words("english"))
special_characters = punctuation.replace("!", "") + string.digits

mail_reg = r'^(\w|\.|\_|\-)+[@](\w|\_|\-|\.)+[.]\w{2,3}$'
url_addresses_reg = r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?|http:/\W"|http:\/\/\w\W\.\W'
retweets_reg = r'^rt[\s]+|rt '

with open(additional_files_dir + "/contractions.json") as f:
    contractions = json.loads(f.read())


def get_file_index(file):
    try:
        file_name, extension = file.split('.')
    except ValueError:
        return None
    
    return int(file_name)


def join_path(*paths):
    return os.path.sep.join(os.path.normpath(path.rstrip("/")) for path in paths)


def move_files(files_names, extension, files_path, destination_path):
    # change the working directory to the directory where we have all the date files of the current dataset
    os.chdir(files_path)
    for file in files_names:
        shutil.move(files_path + '/' + file + extension, destination_path)


def rearrange_text_dataframe(df):
    new_df = pd.concat(df)
    new_df = new_df.reset_index(drop=True) # resetting the index and removing old index
    new_df['text'] = new_df[new_df.columns[:]].apply(lambda x : ' '.join(x.dropna().astype(str)), axis=1)
    new_df = new_df[['text']].astype(str)

    return new_df


def read_line(input_file_path):
    input_file = open(input_file_path, 'r')

    while True:
        line = input_file.readline()
        yield line


def read_image(path):
    try:
        img = cv.resize(cv.imread(path), (128, 128))
        img = img.astype(np.float32) / 255.
    except:
        print('Unexpected error: ', sys.exc_info()[0])
        sys.exit(1)

    return img


def get_images(path, images_filenames):
    images = []
    for image_filename in images_filenames:
        img = read_image(join_path(path, image_filename))
        images.append(img)
    
    images = np.array(images)
    return images


def get_text_dataframe(path, text_filenames):
    df_text = [pd.read_csv(join_path(path, file), header=None, sep=None, engine='python', usecols=None, squeeze=None) for file in text_filenames]
    df_text = rearrange_text_dataframe(df_text)
    
    return df_text


def normalize_text(x):
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore') # remove accented chars
    return x


def expand_words(tweet):
    for key, value in contractions.items():
        tweet = tweet.replace(key, value)
    return tweet


def replace_char(old_str, pos, chr):
    list_chrs = list(old_str)
    list_chrs[pos] = chr
    return ''.join(list_chrs)


"""
    any letter repeated more than three times in a row is replaced by two repetitions of the same letter
"""
def remove_multiple_occurences(text):
    n = len(text)

    if n < 3:
        return text

    i, count = 0, 0
    while i < n - 1:
        i += 1
        if text[i] != text[i-1]:
            count = 0
        else:
            count += 1
            if count >= 2:
                text = text[:i] + text[i+1:]
                n -= 1
                i -= 1

    return text


def remove_stopwords(text):
  return " ".join([word for word in str(text).split() if word not in STOPWORDS])


def split_attached_words(text):
    text = " ".join([groupe for groupe in re.split(r"([A-Z][a-z]+[^A-Z]*)", text) if groupe])
    return text