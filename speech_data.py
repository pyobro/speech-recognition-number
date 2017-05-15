import os
import re
import sys
import wave

import numpy as np
import skimage.io  # scikit-image
import librosa

from random import shuffle
from six.moves import urllib
from six.moves import xrange

# TRAIN_INDEX='train_words_index.txt'
# TEST_INDEX='test_words_index.txt'
SOURCE_URL = 'http://pannous.net/files/'
DATA_DIR = 'data/'
pcm_path = "data/spoken_numbers_pcm/"  # 8 bit
wav_path = "data/spoken_numbers_wav/"  # 16 bit s16le
path = pcm_path
CHUNK = 4096
test_fraction = 0.1  # 10% of data for test / verification

