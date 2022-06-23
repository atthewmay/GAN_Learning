import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm


def gradient_penalty(critic,real,fake,labels,device="cpu"):
    BATCH_SIZE, C,H,W = real.shape
    epsilon = torch.rand((BATCH_SIZE,1,1,1)).repeat(1,C,H,W).to(device)
    interpolated_images = real*epsilon + fake*(1-epsilon)

    mixed_scores = critic(interpolated_images,labels)
    gradient = torch.autograd.grad(
            inputs = interpolated_images,
            outputs = mixed_scores,
            grad_outputs = torch.ones_like(mixed_scores), # If thinking x-->y, graph, this is dL/dy. We don't have a
                                                            # loss here, so we just set to 1 for each dimension of y
            create_graph= True,
            retain_graph = True,
            )[0]
    gradient = gradient.view(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2,dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1)**2)
    return gradient_penalty

def display_embeddings(disc):
    range_vec = [i for i in range(10)]
    embeds = disc.embed(torch.tensor(range_vec)).detach().numpy()
    print(f"shape of embeds is {embeds.shape}")
    ncol = 3
    nrow = math.ceil(len(range_vec)/ncol)
    fig,ax = plt.subplots(ncol,nrow)
    ax = ax.flatten()
    for i in range_vec:
        ax[i].imshow(embeds[i].reshape(int(math.sqrt(embeds[i].shape[0])),-1),cmap='Greys')
    plt.show()


#FID STUFF
class fidCalculator():
    def __init__(self,device='cpu'):
        Iv3 = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True,aux_logits=False)
        Iv3.eval()
        Iv3_truncated = torch.nn.Sequential(*list(Iv3.children())[:-2])
        self.device = device
        self.Iv3_truncated = Iv3_truncated.to(device)
        self.Iv3_truncated.eval()

        #The following assumes MNIST
        self.transforms = transforms.Compose(
                [
                    transforms.Resize(299),
                    transforms.CenterCrop(299),
#                     transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def prep_images(self,img_batch):
        if img_batch.shape[1]==1:
            img_batch = img_batch.expand(-1,3,-1,-1)
        return img_batch


    def calc_FID(self,real,fake):
        real,fake = self.prep_images(real),self.prep_images(fake)
        with torch.no_grad():
            real_vec = self.Iv3_truncated(self.transforms(real)).detach().cpu().numpy().squeeze()
            fake_vec = self.Iv3_truncated(self.transforms(fake)).detach().cpu().numpy().squeeze()
            return self.FID_formula(real_vec,fake_vec)

    def FID_formula(self,real_vec,fake_vec):
        mu_real,mu_fake = np.mean(real_vec,axis=1),np.mean(fake_vec,axis=1)
        cov_real,cov_fake = np.cov(real_vec,rowvar=False), np.cov(fake_vec,rowvar=False)
        matrix_sqrt = sqrtm(np.matmul(cov_real,cov_fake))
        if np.iscomplexobj(matrix_sqrt):
            matrix_sqrt = matrix_sqrt.real
        fid = np.sum((mu_real-mu_fake)**2) + np.trace(cov_real+cov_fake-2*matrix_sqrt)
        return fid


