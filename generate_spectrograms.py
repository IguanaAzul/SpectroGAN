from spectrogram_conversion import generate_all_spectrograms
import matplotlib
import time
matplotlib.use("Agg")


t0 = time.time()
generate_all_spectrograms(
        audio_fpath="./audios/",
        spectrograms_path="./spectrograms/",
        seconds_per_file=20,
        img_size=512,
        replace=False
)
print(f"Tempo para gerar os espectrogramas: {time.time() - t0}")
