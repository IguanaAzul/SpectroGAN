import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as torch_dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import gc
import os
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
seed = 1
random.seed(seed)
torch.manual_seed(seed)


def load_model(load_path, model, z_vector, n_features_generator, n_features_discriminator, n_channels):
    load_path = load_path + f"/model_{model}/"
    generator = Generator(z_vector, n_features_generator, n_channels).to(device)
    discriminator = Discriminator(n_features_discriminator, n_channels).to(device)
    generator.load_state_dict(torch.load(load_path+"generator"))
    discriminator.load_state_dict(torch.load(load_path + "discriminator"))
    return generator, discriminator


def save_model(save_path, generator, discriminator):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    i = 0
    while os.path.exists(save_path + f"/model_{i}/"):
        i += 1
    save_path = save_path + f"/model_{i}/"
    os.mkdir(save_path)
    torch.save(generator, save_path+"generator.pth")
    torch.save(discriminator, save_path + "discriminator.pth")
    torch.save(generator.state_dict(), save_path + "generator_state_dict.pth")
    torch.save(discriminator.state_dict(), save_path + "discriminator_state_dict.pth")


def load_dataset(images_folder_path, image_size, batch_size):
    dataset = torch_dataset.ImageFolder(
        root=images_folder_path, transform=transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ]
        )
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, z_vector, n_features_generator, n_channels):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_vector, n_features_generator * 32, 4, 1, bias=False),
            nn.BatchNorm2d(n_features_generator * 32),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_features_generator * 32, n_features_generator * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features_generator * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_features_generator * 16, n_features_generator * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features_generator * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_features_generator * 8, n_features_generator * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features_generator * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_features_generator * 4, n_features_generator * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features_generator * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_features_generator * 2, n_features_generator, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features_generator),
            nn.ReLU(True),
            nn.ConvTranspose2d(n_features_generator, n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, inputs):
        return self.main(inputs)


class Discriminator(nn.Module):
    def __init__(self, n_features_discriminator, n_channels):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(n_channels, n_features_discriminator, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_features_discriminator, n_features_discriminator * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features_discriminator * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_features_discriminator * 2, n_features_discriminator * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features_discriminator * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_features_discriminator * 4, n_features_discriminator * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features_discriminator * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_features_discriminator * 8, n_features_discriminator * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features_discriminator * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_features_discriminator * 16, n_features_discriminator * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_features_discriminator * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_features_discriminator * 32, 1, 4, 1, 0, bias=False)
        )

    def forward(self, inputs):
        return self.main(inputs)


def train_gan(
        images_folder_path,
        image_size,
        batch_size,
        n_features_discriminator,
        n_features_generator,
        z_vector,
        n_channels,
        num_epochs,
        beta1,
        lr,
        load_params=None,
        save_path="./models/",
        save_every_epoch=False,
        show_epoch_results=True,
):
    dataloader = load_dataset(images_folder_path, image_size, batch_size)
    if load_params:
        generator_net, discriminator_net = load_model(
            load_params["load_path"],
            load_params["model"],
            load_params["epoch"],
            z_vector,
            n_features_generator,
            n_features_discriminator,
            n_channels
        )
    else:
        generator_net = Generator(z_vector, n_features_generator, n_channels).to(device)
        if device.type == 'cuda':
            generator_net = nn.DataParallel(generator_net)
        generator_net.apply(weights_init)
        print(generator_net)

        discriminator_net = Discriminator(n_features_discriminator, n_channels).to(device)
        if device.type == 'cuda':
            discriminator_net = nn.DataParallel(discriminator_net)
        discriminator_net.apply(weights_init)
        print(discriminator_net)

    criterion = nn.BCEWithLogitsLoss()

    fixed_noise = torch.randn(64, z_vector, 1, 1, device=device)

    real_label = 1.
    fake_label = 0.

    optimizer_discriminator = optim.Adam(discriminator_net.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_generator = optim.Adam(generator_net.parameters(), lr=lr, betas=(beta1, 0.999))

    img_list = []
    generator_losses = []
    discriminator_losses = []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            discriminator_net.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = discriminator_net(real_cpu)
            output = output.view(-1)
            err_discriminator_real = criterion(output, label)
            err_discriminator_real.backward()
            x_discriminator = output.mean().item()

            torch.cuda.empty_cache()
            gc.collect()

            noise = torch.randn(b_size, z_vector, 1, 1, device=device)
            fake = generator_net(noise)
            label.fill_(fake_label)
            output = discriminator_net(fake.detach()).view(-1)
            err_discriminator_fake = criterion(output, label)
            err_discriminator_fake.backward()
            disc_gen_z1 = output.mean().item()
            err_discriminator = err_discriminator_real + err_discriminator_fake
            optimizer_discriminator.step()

            torch.cuda.empty_cache()
            gc.collect()

            generator_net.zero_grad()
            label.fill_(real_label)
            output = discriminator_net(fake).view(-1)
            err_generator = criterion(output, label)
            err_generator.backward()
            disc_gen_z2 = output.mean().item()
            optimizer_generator.step()

            torch.cuda.empty_cache()
            gc.collect()

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         err_discriminator.item(), err_generator.item(), x_discriminator, disc_gen_z1, disc_gen_z2))

            generator_losses.append(err_generator.item())
            discriminator_losses.append(err_discriminator.item())

            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = generator_net(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

        if save_every_epoch:
            save_model(save_path, generator_net, discriminator_net)

        if show_epoch_results:
            real_batch = next(iter(dataloader))

            plt.figure(figsize=(15, 15))
            plt.subplot(1, 2, 1)
            plt.axis("off")
            plt.title("Real Images")
            plt.imshow(
                np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),
                             (1, 2, 0))
            )

            plt.subplot(1, 2, 2)
            plt.axis("off")
            plt.title("Fake Images")
            plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
            plt.show()

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(generator_losses, label="G")
    plt.plot(discriminator_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    save_model(save_path, generator_net, discriminator_net)
