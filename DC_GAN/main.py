import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import models as m

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1

Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)],[0.5 for _ in range(CHANNELS_IMG)]),
        ]
    )

dataset = datasets.MNIST(root="dataset/",train=True,transform=transforms,download=True)

loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
gen = m.Generator(Z_DIM,CHANNELS_IMG,FEATURES_GEN).to(device)
disc = m.Discriminator(CHANNELS_IMG,FEATURES_DISC).to(device)
m.initialize_weights(gen)
m.initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(),lr = LEARNING_RATE,betas = (0.5,0.999))
opt_disc = optim.Adam(disc.parameters(),lr = LEARNING_RATE,betas = (0.5,0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32,Z_DIM,1,1).to(device)

writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real,_) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE,Z_DIM,1,1)).to(device)
        fake = gen(noise)
        #
        disc_real = disc(real).reshape(-1)
        disc_loss_real = criterion(disc_real,torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        disc_loss_fake = criterion(disc_fake,torch.ones_like(disc_fake))
        loss_disc = (disc_loss_real+disc_loss_fake)/2
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        output = disc(fake).reshape(-1)
        loss_gen = criterion(output,torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx%100 == 0:
            print(f"Losses at epoch: {epoch}, batch: {batch_idx} ==> \n\t Discriminator = {loss_disc} \n\t Generator = {loss_gen}")

            with torch.no_grad():
                fake = gen(fixed_noise)
                grid_real = torchvision.utils.make_grid(real[:32],normalize=True)
                grid_fake = torchvision.utils.make_grid(fake[:32],normalize=True)
                writer_real.add_image("real",grid_real,global_step = step)
                writer_fake.add_image("fake",grid_fake,global_step = step)
        step += 1



