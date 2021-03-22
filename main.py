from audio_to_spectrogram import generate_all_spectrograms

seconds_per_file = 20
file_size = 256
audio_fpath = "./audios/"
spectrograms_path = "./spectrograms/"

generate_all_spectrograms(audio_fpath, spectrograms_path, seconds_per_file, file_size)


# TODO: Treinamento da GAN

# TODO: Implementar transformação de espectrograma para audio
