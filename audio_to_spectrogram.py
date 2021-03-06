import os
from matplotlib import pyplot as plt
import matplotlib
import librosa
import librosa.display
import IPython.display as ipd
from argparse import ArgumentParser
matplotlib.use('Agg')

def generate_spectrogram(x, sr, save_name):
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    fig = plt.figure(figsize=(256, 256), dpi=32, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.axis('off')
    librosa.display.specshow(Xdb, sr=sr, cmap='grey')
    plt.savefig(save_name, bbox_inches=0, pad_inches=0)
    plt.close()
    librosa.cache.clear()

parser = ArgumentParser()
parser.add_argument("-n", "--file_number", dest="file_number")
args = parser.parse_args()

file_number = int(args.file_number)
audio_fpath = "./audios/"
spectrograms_path = "./spectrograms/"
audio_clips = os.listdir(audio_fpath)

clip = audio_clips[file_number]
#for idx, i in enumerate(audio_clips):
#    print(idx, i)
print(clip)

audio_length = librosa.get_duration(filename=audio_fpath + clip)
j=60
while j < audio_length:
    x, sr = librosa.load(audio_fpath + clip, offset=j-60, duration=60)
    save_name = spectrograms_path + clip + str(j) + ".png"
    generate_spectrogram(x, sr, save_name)
    j += 60
    if j >= audio_length:
        j = audio_length
        x, sr = librosa.load(audio_fpath + clip, offset=j-60, duration=60)
        save_name = spectrograms_path + clip + str(j) + ".png"
        generate_spectrogram(x, sr, save_name)
