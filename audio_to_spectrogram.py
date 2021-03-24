from matplotlib import pyplot as plt
import librosa
import librosa.display
import os


def generate_spectrogram(x, sr, save_name, file_size):
    x_transformed = librosa.stft(x)
    xdb = librosa.amplitude_to_db(abs(x_transformed))
    fig = plt.figure(figsize=(file_size, file_size), dpi=1, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.axis('off')
    librosa.display.specshow(xdb, sr=sr, cmap='gray')
    plt.savefig(save_name, bbox_inches=0, pad_inches=0)
    plt.close()
    librosa.cache.clear()


def generate_all_spectrograms(
        audio_fpath="./audios/",
        spectrograms_path="./spectrograms/",
        seconds_per_file=20,
        file_size=256,
        replace=False
):
    audio_clips = os.listdir(audio_fpath)
    folders = [spectrograms_path + clip.split(".", 1)[0] + "/" for clip in audio_clips]

    for folder in folders:
        if not os.path.exists(folder):
            os.mkdir(folder)

    for clip, folder in zip(audio_clips, folders):
        audio_length = librosa.get_duration(filename=audio_fpath + clip)
        j = seconds_per_file
        while j < audio_length:
            x, sr = librosa.load(audio_fpath + clip, offset=j - seconds_per_file, duration=seconds_per_file)
            save_name = folder + str(j) + ".png"
            if not os.path.exists(save_name) or replace:
                generate_spectrogram(x, sr, save_name, file_size)
            j += seconds_per_file
            if j >= audio_length:
                j = audio_length
                x, sr = librosa.load(audio_fpath + clip, offset=j - seconds_per_file, duration=seconds_per_file)
                save_name = folder + str(j) + ".png"
                if not os.path.exists(save_name) or replace:
                    generate_spectrogram(x, sr, save_name, file_size)
