import os
from matplotlib import pyplot as plt
import matplotlib
import librosa
import librosa.display
matplotlib.use('Agg')

seconds_per_file = 20
file_size = 512

def generate_spectrogram(x, sr, save_name):
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    fig = plt.figure(figsize=(file_size, file_size), dpi=1, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.axis('off')
    librosa.display.specshow(Xdb, sr=sr, cmap='gray')
    plt.savefig(save_name, bbox_inches=0, pad_inches=0)
    plt.close()
    librosa.cache.clear()

audio_fpath = "./audios/"
spectrograms_path = "./spectrograms/"
audio_clips = os.listdir(audio_fpath)
folders = [spectrograms_path + clip.split(".", 1)[0] + "/" for clip in audio_clips]

for folder in folders:
    if not os.path.exists(folder):
        os.mkdir(folder)

for clip, folder in zip(audio_clips, folders):
    audio_length = librosa.get_duration(filename=audio_fpath + clip)
    j=seconds_per_file
    while j < audio_length:
        x, sr = librosa.load(audio_fpath + clip, offset=j-seconds_per_file, duration=seconds_per_file)
        save_name = folder + str(j) + ".png"
        if not os.path.exists(save_name):
            generate_spectrogram(x, sr, save_name)
        j += seconds_per_file
        if j >= audio_length:
            j = audio_length
            x, sr = librosa.load(audio_fpath + clip, offset=j-seconds_per_file, duration=seconds_per_file)
            save_name = folder + str(j) + ".png"
            if not os.path.exists(save_name):
                generate_spectrogram(x, sr, save_name)
