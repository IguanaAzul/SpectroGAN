from audio_to_spectrogram import generate_all_spectrograms
import matplotlib
matplotlib.use("Agg")

generate_all_spectrograms(
        audio_fpath="./audios/",
        spectrograms_path="./spectrograms/",
        seconds_per_file=20,
        img_size=512,
        replace=False
)
