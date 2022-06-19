import torch
import torch.nn as nn
import matplotlib.pyplot as plt


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
    embeds = disc.embed(torch.tensor(range_vec))
    print(f"shape of embeds is {embeds.shape}")
    ncols = 3
    nrow = math.ceil(len(range_vec)/ncols)
    fig,ax = plt.subplots(ncols,nrows)
    ax = ax.flatten()
    for i in range_vec:
        ax[i].imshow(embeds[i].reshape(8,-1),cmap='grey')
    plt.show()

