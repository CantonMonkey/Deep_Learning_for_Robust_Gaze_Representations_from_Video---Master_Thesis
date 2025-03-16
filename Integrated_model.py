


## layer 1, understand strides and kernel_size concepts, we do not freeze the layers i think? maybe if there is overfitting we can freeze resnet for less training parameters
# TODO: Ask about copying codes from documentation
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
# https://flypix.ai/blog/image-recognition-algorithms/ why I choosed only first few layers for feature extraction, upto middle layers are sufficient to provide information about eyes
# In higher layers of the network, detailed pixel information is lost whilethe high level content of the image is preserved. Clear Explanation is here(https://ai.stackexchange.com/questions/30038/why-do-we-lose-detail-of-an-image-as-we-go-deeper-into-a-convnet)
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        print(resnet)

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-3])
        
        
        self.feature_extractor[-1][0].conv1.stride = (1, 1) # normally resnet has stride = 2 in layer 3 this will downsample from 8x8 to 4x4. This is to avoid this. More spatil features are preserved
        self.feature_extractor[-1][0].downsample[0].stride = (1, 1)

        self.reduce_channels = nn.Conv2d(256, 128, kernel_size=1) # kernel_size 1 only changes the number of channels and doesn't mess with the spatial size

        #layer 2:

    def forward(self, x):
        x = self.feature_extractor(x)  
        x = self.reduce_channels(x)   
        return x

feature_extractor = ResNetFeatureExtractor()
feature_extractor.eval()  

transform = transforms.Compose([ # pre processing the images to match the resnet's training statistics
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalization of the pixel values
])



class FeatureFusion(torch.nn.Module):
    def __init__(self, total_ch = 384):    # the gaze_dims should be defined later, according to the dataset and what we wanna predict, for instance, 3 gaze dims, PoG (x,y,z)
        super(FeatureFusion, self).__init__()
        # total_ch = 384
        self.gn = torch.nn.GroupNorm(3, total_ch)     # Separate 6 channels into 3 groups, how to define a appropriate number of groups & channels?
    def forward(self, left_eye, right_eye, face):
        # layer2
        # layer 2: feature fusion: concate + group  normalization
        # self.concate = torch.cat((x, x, x), 0) // not needed here, only in forward
        # total_ch = Leye[1]+Reye[1]+FaceData[1]  # total channels of the input // this is not working 
        concate = torch.cat((left_eye, right_eye, face), 1)  # dim = 0 or 1?  only channel dim changes?
        out = self.gn(concate)
        return out

# from vit_pytorch import ViT
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
from vit_pytorch.vit import Transformer


class Attention(torch.nn.Module):
    def __init__(self, total_ch = 384):    # the gaze_dims should be defined later, according to the dataset and what we wanna predict, for instance, 3 gaze dims, PoG (x,y,z)
        super(Attention, self).__init__()
        self.self_att = Transformer(
                dim = total_ch,   # if not using the total_ch, should project the input to the dim of the transformer first
                depth = 6,
                heads = 16,
                dim_head = total_ch//16,  #dim//heads
                mlp_dim = 2048,  # the hidden layer dim of the mlp (the hidden layer of the feedforward network, which is applied to each position (each token) separately and identically)
                # dropout = 0.
                # def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.)
                
        )


    def forward(self):
        bs, c, h, w = self.out.shape
        x_att = self.out.reshape(bs, c, h * w).transpose(1, 2)   # (bs, h*w, c) --- (bs, seq_len, features)
        x_att = self.self_att(x_att)  # output shape (bs, h*w, c)

        #reshape the output
        # for input frames, the temporal information should be considered, how to define the input_size and hidden_size?
        return x_att
        

        # set learning rate for diff layers
     


class Temporal(torch.nn.Module):
    def __init__(self, total_ch = 384):    # the gaze_dims should be defined later, according to the dataset and what we wanna predict, for instance, 3 gaze dims, PoG (x,y,z)
        super(Temporal, self).__init__()
        # RNN layer for temporal information
        # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
        # torch.nn.GRU(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)
        # how to define the input_size and hidden_size?
        # test num_layers param, what does it affect?
        self.rnn = torch.nn.GRU(total_ch, 512, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)

    def forward(self, x_att, h_state):
        # h_n itself should be an input for GRU, otherwise useless, dont forget the hidden state
        out, h_n = self.rnn(x_att, h_state) # read the source, there are 2 outputs, but what is h_n here? should be the hidden state of the last layer?
        # mind the coherence of the input and output of the RNN layer 
        
        #reshape the output
        # for input frames, the temporal information should be considered, how to define the input_size and hidden_size?
        return out, h_n
        

        # set learning rate for diff layers




class GazePrediction(nn.Module):
    def __init__(self):
        super(GazePrediction, self).__init__()
        self.fc = nn.Linear(384, 3)  
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  
        x = self.fc(x) 
        return x

