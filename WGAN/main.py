import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import models as m
import argparse

save_path = "saved_models/"
parser = argparse.ArgumentParser()
parser.add_argument('--resume_training',action="store_true")
parser.add_argument('--save_models',action="store_true")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
LEARNING_RATE = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1

Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

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
critic = m.Discriminator(CHANNELS_IMG,FEATURES_DISC).to(device)
if args.resume_training:
    gen.load_state_dict(torch.load(save_path+"generator_model"))
    critic.load_state_dict(torch.load(save_path+"criticriminator_model"))
m.initialize_weights(gen)
m.initialize_weights(critic)

opt_gen = optim.RMSprop(gen.parameters(),lr = LEARNING_RATE)
opt_critic = optim.RMSprop(critic.parameters(),lr = LEARNING_RATE)

fixed_noise = torch.randn(32,Z_DIM,1,1).to(device)

writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real,_) in enumerate(loader):
        real = real.to(device)
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((BATCH_SIZE,Z_DIM,1,1)).to(device)
            fake = gen(noise)
            #
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
#         critic_loss_real = criterion(critic_real,torch.ones_like(critic_real))
#         critic_loss_fake = criterion(critic_fake,torch.zeros_like(critic_fake))
            loss_critic = - (torch.mean(critic_real) - torch.mean(critic_fake))
            critic.zero_grad() # Ah, right ,else would erase intermediate states from Generator.
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP,WEIGHT_CLIP)

## Train gen
        output = critic(fake).reshape(-1)
        loss_gen = - torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx%100 == 0:
            print(f"Losses at epoch: {epoch}, batch: {batch_idx} ==> \n\t Discriminator = {loss_critic} \n\t Generator = {loss_gen}")

            with torch.no_grad():
                fake = gen(fixed_noise)
                grid_real = torchvision.utils.make_grid(real[:32],normalize=True)
                grid_fake = torchvision.utils.make_grid(fake[:32],normalize=True)
                writer_real.add_image("real",grid_real,global_step = step)
                writer_fake.add_image("fake",grid_fake,global_step = step)
            if args.save_models:
                torch.save(critic.state_dict(),save_path+"criticriminator_model")
                torch.save(gen.state_dict(),save_path+"generator_model")
        step += 1



