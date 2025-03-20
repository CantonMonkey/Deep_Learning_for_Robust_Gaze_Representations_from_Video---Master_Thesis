


## layer 1, understand strides and kernel_size concepts, we do not freeze the layers i think? maybe if there is overfitting we can freeze resnet for less training parameters
# TODO: Ask about copying codes from documentation
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

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


    def forward(self, x):
        x = self.feature_extractor(x)  
        x = self.reduce_channels(x)   
        print("ResNetFeatureExtractor", x.shape)
        return x  # (bs, ch, h, w)




class FeatureFusion(torch.nn.Module):
    """
    Feature Fusion Layer
    input shape: (bs, ch, h, w)
    output shape: (bs, ch, h, w)
    """
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
        print("FeatureFusion", out.shape)
        return out 

# from vit_pytorch import ViT
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
from vit_pytorch.vit import Transformer


class Attention(torch.nn.Module):
    """
    Attention Layer
    class input shape: (bs, ch, h, w)
    in forward, reshape to (bs, h*w, ch)
    output shape: (bs, h*w, ch)
    """
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


    def forward(self, out):
        bs, c, h, w = out.shape
        x_att = out.reshape(bs, c, h * w).transpose(1, 2)   # (bs, h*w, c) --- (bs, seq_len, features)
        x_att = self.self_att(x_att)  # output shape (bs, h*w, c)

        #reshape the output
        # for input frames, the temporal information should be considered, how to define the input_size and hidden_size?

        print("Attention", x_att.shape)
        return x_att
        

        # set learning rate for diff layers
     


class Temporal(torch.nn.Module):
    """
    Temporal Layer
    input shape: (bs, ch, h, w)
    output shape: (seq_len, bs, ch)  # seq_len = h*w 
    h_n shape: (num_layers, bs, hidden_size)
    """
    def __init__(self, total_ch = 384):    # the gaze_dims should be defined later, according to the dataset and what we wanna predict, for instance, 3 gaze dims, PoG (x,y,z)
        super(Temporal, self).__init__()
        # RNN layer for temporal information
        # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
        # torch.nn.GRU(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)
        # how to define the input_size and hidden_size?
        # test num_layers param, what does it affect?
        self.gru = torch.nn.GRU(input_size=total_ch, 
                                hidden_size=512,    # the more hidden size, the complexer memory
                                num_layers=5,       # does this represent the number of GRU blocks? or say the number of consecutive frames we wanna consider?
                                bias=True, 
                                batch_first=False, 
                                dropout=0.0, 
                                bidirectional=False, 
                                device=None, 
                                dtype=None)
        
        # do the params need to be defined in constructor?

        # input of GRU:
        # :math:`(L, N, H_{in})` when ``batch_first=False`` or
        #   :math:`(N, L, H_{in})` when ``batch_first=True``
        # for the output of transformer, the h*w represents the seq_length, and the total_ch represents the features (which is H_{in} or input size)
        # output of transformer : (ba, h*w, ch), should be reshaped to (seq_len, bs, features) for GRU, which is (h*w, bs, total_ch)
        # the transformation is done in the Class "WholeModel"
    
        

    def forward(self, x_att, h_state=None):
        print("Temporal_start", x_att.shape, h_state)
        # h_n itself should be an input for GRU, otherwise useless, dont forget the hidden state
        out, h_n = self.gru(x_att, h_state) # read the source, there are 2 outputs, but what is h_n here? should be the hidden state of the last layer?
        # mind the coherence of the input and output of the RNN layer 
        
        #reshape the output
        # for input frames, the temporal information should be considered, how to define the input_size and hidden_size?

        print("Temporal out", out.shape)
        print("Temporal h_n", h_n.shape)
        return out, h_n   # okay one important thing is to use h_n or out for fc layer?
        # answer: use out, since it contains the info of all "time steps" (or frame steps). However, h_n only contains the info of the last time step. 
        

        # set learning rate for diff layers



class GazePrediction(nn.Module):
    """
    FC layer
    input shape: (seq_len, bs, ch)  # output of GRU 
    reshape input to ( bs, seq_len*ch)  # flatten the input
    output shape: (bs, num_classes)  # output of the model
    """
    def __init__(self, hidden_size, sequence_length, num_classes):
        super(GazePrediction, self).__init__()
        # do we need more linear layers and dense/dropout before label mapping?
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)  #https://www.kaggle.com/code/fanbyprinciple/learning-pytorch-3-coding-an-rnn-gru-lstm
        # self.fc = nn.Linear(384, 3)   # I suppose it should be mapped the output of the GRU to the h_n.
    
    def forward(self, x):
        print ("enter FC layer")
        x = x.view(x.size(0), -1)   # why??  # x.size represents the bs, and -1 will flatten the rest dims to 1D vector, it turns out to be (bs, features).)
        # one question: for regression task, also use nn.Linear to map them to the output?
        x = self.fc(x)              # after conventing to 1D vector, the output of the fc layer is (bs, num_classes) 
        print("GazePrediction", x.shape)
        return x




class WholeModel(nn.Module):
    def __init__(self):
        super(WholeModel, self).__init__()
        self.layers= (nn.ModuleList([
                ResNetFeatureExtractor(),
                FeatureFusion(),
                Attention(),
                Temporal(),
                GazePrediction(hidden_size=512, sequence_length=16, num_classes=3)  # why sequence_length = 64? h*w*bs, yes indeed :)
                # RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x8192 and 32768x3) from 8192/512=16 I know seq_len = 16, but why?
                # answer: read the source code of GRU, the output of GRU is (seq_len, bs, hidden_size), so the input of FC layer should be (bs, seq_len*hidden_size)
            ]))
        
        # self.left_eye = self.layers[0]
        # self.right_eye = self.layers[0]
        # self.face = self.layers[0]
        

    def forward(self, left_eye, right_eye, face):
        # calculate 3 input features in parallel
        left_eye = self.layers[0](left_eye)
        right_eye = self.layers[0](right_eye)
        face = self.layers[0](face)
        ##################################

        fusioned_feature = self.layers[1](left_eye, right_eye, face)

        Attention_map = self.layers[2](fusioned_feature)

        bs, seq_len, ch = Attention_map.shape
        Attention_map = Attention_map.reshape(seq_len, bs, ch)

        gru_out, _ = self.layers[3](Attention_map)  # h_n is not needed for FC layer, or it can be used if other tequniques are used 
        # so, I think multiple GRU blocks are needed. Answer: No, just change the num_layers param in the GRU block
        # gru_out = gru_out.reshape(gru_out.shape[0], -1) ##https://www.kaggle.com/code/fanbyprinciple/learning-pytorch-3-coding-an-rnn-gru-lstm # alredy done in the FC class

        # reshape the output of GRU to (bs, seq_len, hidden_size)
        # seq_len, bs, ch = gru_out.shape
        # gru_out = gru_out.reshape(bs, seq_len, ch)  # (bs, seq_len*hidden_size)
        gru_out = gru_out.reshape(gru_out.shape[0], -1)   
        print("gru_out", gru_out.shape)

        pred = self.layers[4](gru_out)
        print("WholeModel", pred.shape)
        return pred

