import librosa.display
import matplotlib.pyplot as plt
import glob

audio_tr_file_list = glob.glob('audio_data/train/*.wav')
audio_te_file_list = glob.glob('audio_data/test/*.wav')

# load all training files of train directory

for i in range(len(audio_tr_file_list) + 1):
    y, sr = librosa.load(audio_tr_file_list[i])
    librosa.feature.mfcc(y=y, sr=sr)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    librosa.feature.mfcc(S=librosa.power_to_db(S))

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    plt.figure(figsize=(5, 5))
    librosa.display.specshow(mfccs)
    # save the graph in the img_data directory
    plt.savefig(
        "img_data/train/img." + audio_tr_file_list[i].split('/')[2].split('.')[0] +
        audio_tr_file_list[i].split('/')[2].split('.')[1] +
        audio_tr_file_list[i].split('/')[2].split('.')[2] + ".jpg")

# load all test files of test directory

for i in range(len(audio_te_file_list) + 1):
    y, sr = librosa.load(audio_te_file_list[i])
    librosa.feature.mfcc(y=y, sr=sr)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    librosa.feature.mfcc(S=librosa.power_to_db(S))

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    plt.figure(figsize=(5, 5))
    librosa.display.specshow(mfccs)
    # save the graph in the img_data directory
    plt.savefig(
        "img_data/test/" + str(i) + ".jpg")
