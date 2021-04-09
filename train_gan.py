from dcgan import train_gan, save_model
import time
import torch
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Seta as variáveis
seconds_per_file = 20
image_size = 512
spectrograms_path = "./spectrograms/"
models_folder = "./models/model3/"
batch_size = 8
n_channels = 1
z_vector = 128
n_features_generator = 32
n_features_discriminator = 32
num_epochs = 10000
lr = 0.0001
beta1 = 0.5

# Caso queira continuar o treinamento de onde parou:
# generator = torch.load("./models/model3/model_42/generator.pth")
# discriminator = torch.load("./models/model3/model_42/discriminator.pth")

print("Treinando DCGAN")
t0 = time.time()
generator, discriminator = train_gan(
        spectrograms_path, image_size, batch_size,
        n_features_discriminator, n_features_generator,
        z_vector, n_channels, num_epochs, beta1, lr,
        save_every_epoch=True, seconds_per_file=seconds_per_file,
        models_folder=models_folder
        # Caso queira continuar o treinamento de onde parou: , generator=generator, discriminator=discriminator
)

print(f"Tempo para treinar o modelo: {time.time() - t0}")

save_model(
        models_folder,
        generator,
        discriminator,
        seconds_per_file,
        image_size,
        batch_size,
        n_channels,
        z_vector,
        n_features_generator,
        n_features_discriminator,
        num_epochs,
        lr,
        beta1,
)
