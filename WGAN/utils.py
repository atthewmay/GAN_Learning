import torch
import torch.nn as nn

def gradient_penalty(critic,real,fake,device="cpu"):
    BATCH_SIZE, C,H,W = real.shape
    epsilon = torch.rand((BATCH_SIZE,1,1,1)).repeat(1,C,H,W).to(device)
    interpolated_images = real*epsilon + fake*(1-epsilon)

    mixed_scores = critic(interpolated_images)
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

