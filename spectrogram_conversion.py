from matplotlib import pyplot as plt
import numpy as np
import librosa
import librosa.display
import soundfile
import os


def map_values(x, left_min, left_max, right_min, right_max):
    return right_min + ((x - left_min) / (left_max - left_min) * (right_max - right_min))


def unmap_values(y, left_min, left_max, right_min, right_max):
    return (y * (left_min - left_max) + left_max * right_min - left_min * right_max) / (right_min - right_max)


def amp_to_log(x, v_min=0.001, left_min=0, left_max=233.95848, right_min=0, right_max=255):
    left_min = np.log10(np.abs(left_min) + v_min)
    left_max = np.log10(np.abs(left_max) + v_min)
    return map_values(np.log10(np.abs(x) + v_min), left_min, left_max, right_min, right_max)


def log_to_amp(x, v_min=0.001, left_min=0, left_max=233.95848, right_min=0, right_max=255):
    left_min = np.log10(np.abs(left_min) + v_min)
    left_max = np.log10(np.abs(left_max) + v_min)
    return 10 ** unmap_values(x, left_min, left_max, right_min, right_max) - v_min


def calculate_proper_frame_and_hop_size(x, image_size):
    frame = 2 * (image_size - 1)
    hop = (-2 * image_size + x.shape[0] + 2) / (image_size + 1)
    return round(frame), round(hop)


def generate_spectrogram(read_path, write_path, img_size, offset, audio_duration, replace):
    if not os.path.exists(write_path) or replace:
        x, _ = librosa.load(read_path, offset=offset, duration=audio_duration, sr=22050)
        frame_size, hop_size = calculate_proper_frame_and_hop_size(x, img_size)
        x_transformed = np.flip(librosa.stft(x, frame_size, hop_size), axis=0)
        xdb = np.round(amp_to_log(x_transformed)).astype(int)
        fig = plt.figure(figsize=(img_size, img_size), dpi=1, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.axis('off')
        fig.add_axes(ax)
        ax.imshow(xdb, cmap='gray', aspect='auto')
        plt.savefig(write_path, dpi=1, pil_kwargs={"quality": 100})
        plt.show()
        librosa.cache.clear()
        plt.close()


def retrieve_audio(read_path, write_path, sr=22050, repeats=4):
    img = plt.imread(read_path)
    img = np.flip(img[:, :, 0], axis=0)
    img_amp = log_to_amp(img)
    img_amp = np.repeat(img_amp, repeats, axis=1)
    result = librosa.griffinlim(np.abs(img_amp))
    soundfile.write(write_path, result, sr)


def retrieve_audios_from_folder(read_path, write_path, sr=22050, repeats=4, replace=False):
    if not os.path.exists(write_path):
        os.mkdir(write_path)
    for file in os.listdir(read_path):
        if not os.path.exists(f"{write_path}/{file}.wav") or replace:
            retrieve_audio(f"{read_path}/{file}", f"{write_path}/{file}.wav", sr, repeats)


def generate_all_spectrograms(
        audio_fpath="./audios/",
        spectrograms_path="./spectrograms/",
        seconds_per_file=20,
        img_size=512,
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
            generate_spectrogram(audio_fpath + clip, f"{folder}{j}.tiff",
                                 img_size, j - seconds_per_file, seconds_per_file, replace)
            j += seconds_per_file
            if j >= audio_length:
                j = audio_length
                generate_spectrogram(audio_fpath + clip, f"{folder}{j}.tiff",
                                     img_size, j - seconds_per_file, seconds_per_file, replace)
