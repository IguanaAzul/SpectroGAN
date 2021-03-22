import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as torch_dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

seed = 1
print("Random Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)
images_folder_path = "./spectrograms/"

batch_size = 1
image_size = 256
n_channels = 1
z_vector = 100
n_features_generator = 32
n_features_discriminator = 32
num_epochs = 5
lr = 0.0002
beta1 = 0.5

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

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_vector, n_features_generator * 8, 4, 1, bias=False),
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

# Convolutional Layer Output Shape = [(W−K+2P)/S]+1
# W is the input volume
# K is the Kernel size
# P is the padding
# S is the stride


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(n_channels, n_features_discriminator, 4, 2, 1, bias=False),
            # input_shape=[256x256], output_shape=[128x128]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_features_discriminator, n_features_discriminator * 2, 4, 2, 1, bias=False),
            # input_shape=[128x128], output_shape=[64x64]
            nn.BatchNorm2d(n_features_discriminator * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_features_discriminator * 2, n_features_discriminator * 4, 4, 2, 1, bias=False),
            # input_shape=[64x64], output_shape=[32x32]
            nn.BatchNorm2d(n_features_discriminator * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_features_discriminator * 4, n_features_discriminator * 8, 4, 2, 1, bias=False),
            # input_shape=[32x32], output_shape=[16x16]
            nn.BatchNorm2d(n_features_discriminator * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_features_discriminator * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, inputs):
        return self.main(inputs)


netG = Generator().to(device)
if device.type == 'cuda':
    netG = nn.DataParallel(netG)
netG.apply(weights_init)
print(netG)

netD = Discriminator().to(device)
if device.type == 'cuda':
    netD = nn.DataParallel(netD)
netD.apply(weights_init)
print(netD)

criterion = nn.BCEWithLogitsLoss()

fixed_noise = torch.randn(64, z_vector, 1, 1, device=device)

real_label = 1.
fake_label = 0.

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_cpu)
        print(output.shape)
        print(label.shape)
        print(label)
        output = output.view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, z_vector, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())

real_batch = next(iter(dataloader))

plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()
