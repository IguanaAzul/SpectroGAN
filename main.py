from audio_to_spectrogram import generate_all_spectrograms
from dcgan import train_gan

# Seta as variáveis
seconds_per_file = 20
image_size = 256
audio_fpath = "./audios/"
spectrograms_path = "./spectrograms/"
batch_size = 16
n_channels = 1
z_vector = 128
n_features_generator = 32
n_features_discriminator = 32
num_epochs = 150
lr = 0.0002
beta1 = 0.5

print("Gerando Espectrogramas")
# generate_all_spectrograms(audio_fpath, spectrograms_path, seconds_per_file, image_size, replace=False)

print("Treinando DCGAN")
train_gan(spectrograms_path, image_size, batch_size,
          n_features_discriminator, n_features_generator,
          z_vector, n_channels, num_epochs, beta1, lr,
          save_every_epoch=True
          )
