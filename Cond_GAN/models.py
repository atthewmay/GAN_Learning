import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,channels_img,features_d,num_classes,img_size):
        super(Discriminator,self).__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
                    nn.Conv2d(channels_img+1,features_d,kernel_size=4,stride=2,padding=1), # add 1 for embedding
                    nn.LeakyReLU(0.2),
                    self._block(features_d, features_d*2, kernel_size=4, stride=2, padding=1),
                    self._block(features_d*2, features_d*4, kernel_size=4, stride=2, padding=1),
                    self._block(features_d*4, features_d*8, kernel_size=4, stride=2, padding=1),
                    nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),
                )
        self.embed = nn.Embedding(num_classes, img_size*img_size)
        # pretty sweet. We are learning the embedding as well. better than just some image_size*img_size w/ a scalar
        # value, bc we can optimize each with different values. It's possible that you'd get a grid looking like a
        # number! I wanna visualize these embeddings I think! So cool. 

    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
                    nn.Conv2d(
                        in_channels,out_channels,kernel_size,stride,padding,bias=False),
                    nn.InstanceNorm2d(out_channels,affine=True),
                    nn.LeakyReLU(0.2),
                )

    def forward(self,x,labels=torch.tensor([1])):
        import pdb; pdb.set_trace()
        embedding = self.embed(labels).view(labels.shape[0],1,self.img_size,self.img_size)
        x = torch.cat([x,embedding],dim = 1)
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self,z_dim,channels_img,features_g,num_classes,embed_size):
        super(Generator,self).__init__()
        self.net = nn.Sequential(
                self._block(z_dim+embed_size,features_g*16,4,1,0),
                self._block(features_g*16,features_g*8,4,2,1),
                self._block(features_g*8,features_g*4,4,2,1),
                self._block(features_g*4,features_g*2,4,2,1),
                nn.ConvTranspose2d(features_g*2,channels_img,kernel_size=4, stride=2, padding=1),
                nn.Tanh(),
        )
        self.embed = nn.Embedding(num_classes,embed_size)


    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,out_channels,kernel_size,stride,padding,bias=False
                    ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
    def forward(self,x,labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x,embedding],dim=1)
        return self.net(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data,0.0,0.02)

def test():
    N,in_channels,H,W = 8,3,64,64
    z_dim = 100
    x = torch.randn((N,in_channels,H,W))
    disc = Discriminator(in_channels,8)
    initialize_weights(disc)
    assert disc(x).shape == (N,1,1,1)

    z = torch.randn((N,z_dim,1,1))
    gen = Generator(z_dim,in_channels,8)
    initialize_weights(gen)
    assert gen(z).shape == (N,in_channels,H,W)

