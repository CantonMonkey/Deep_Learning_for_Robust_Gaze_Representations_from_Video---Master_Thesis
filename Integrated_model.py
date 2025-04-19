## layer 1, understand strides and kernel_size concepts, we do not freeze the layers i think? maybe if there is overfitting we can freeze resnet for less training parameters
# TODO: Ask about copying codes from documentation
import torch
import torch.nn as nn
import torchvision.models as models

# https://flypix.ai/blog/image-recognition-algorithms/ why I choosed only first few layers for feature extraction, upto middle layers are sufficient to provide information about eyes
# In higher layers of the network, detailed pixel information is lost whilethe high level content of the image is preserved. Clear Explanation is here(https://ai.stackexchange.com/questions/30038/why-do-we-lose-detail-of-an-image-as-we-go-deeper-into-a-convnet)
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-3])
        
        self.feature_extractor[-1][0].conv1.stride = (1, 1) # normally resnet has stride = 2 in layer 3 this will downsample from 8x8 to 4x4. This is to avoid this. More spatil features are preserved
        self.feature_extractor[-1][0].downsample[0].stride = (1, 1) 

        self.reduce_channels = nn.Conv2d(256, 128, kernel_size=1) # kernel_size 1 only changes the number of channels and doesn't mess with the spatial size


    def forward(self, x):
        x = self.feature_extractor(x)  
        x = self.reduce_channels(x)   
        #print("ResNetFeatureExtractor", x.shape)
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
        self.gn = torch.nn.GroupNorm(24, total_ch)     # Separate 6 channels into 3 groups, how to define a appropriate number of groups & channels?
    def forward(self, left_eye, right_eye, face):
        # layer2
        # layer 2: feature fusion: concate + group  normalization
        # self.concate = torch.cat((x, x, x), 0) // not needed here, only in forward
        # total_ch = Leye[1]+Reye[1]+FaceData[1]  # total channels of the input // this is not working 
        concate = torch.cat((left_eye, right_eye, face), 1)  # dim = 0 or 1?  only channel dim changes?
        out = self.gn(concate)
        #print("FeatureFusion", out.shape)
        return out 

# from vit_pytorch import ViT
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
from vit_pytorch.vit import Transformer


class Attention(torch.nn.Module):
    """
    Attention Layer
    seq_len = h*w (number of patch = 1, hence token = h*w), features = ch
    class input shape: (bs, ch, h, w)
    in forward, reshape to (bs, h*w, ch)
    output shape: (bs, h*w, ch)
    """
    def __init__(self, total_ch = 384):    # the gaze_dims should be defined later, according to the dataset and what we wanna predict, for instance, 3 gaze dims, PoG (x,y,z)
        super(Attention, self).__init__()
        self.self_att = Transformer(
                dim = total_ch,   # if not using the total_ch, should project the input to the dim of the transformer first
                depth = 3,
                heads = 8,
                dim_head = total_ch//8,  #dim//heads
                mlp_dim = 1024,  # the hidden layer dim of the mlp (the hidden layer of the feedforward network, which is applied to each position (each token) separately and identically)
                dropout = 0.1
                # def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.)
        )


    def forward(self, out):
        bs, c, h, w = out.shape
        x_att = out.reshape(bs, c, h * w).transpose(1, 2)   # (bs, h*w, c) --- (bs, seq_len, features)
        x_att = self.self_att(x_att)  # output shape (bs, h*w, c)

        #reshape the output
        # for input frames, the temporal information should be considered, how to define the input_size and hidden_size?

        #print("Attention", x_att.shape)
        return x_att
        

        # set learning rate for diff layers
     


class Temporal(torch.nn.Module):
    """
    Temporal Layer
    input shape: (bs, h*w, ch)    # ==(bs, seq_len, ch)
    output shape: (bs, seq_len, ch)  # seq_len = h*w 
    h_n shape: (num_layers, bs, hidden_size)
    """
    def __init__(self, total_ch = 384):    # the gaze_dims should be defined later, according to the dataset and what we wanna predict, for instance, 3 gaze dims, PoG (x,y,z)
        super(Temporal, self).__init__()
        # RNN layer for temporal information
        # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
        # torch.nn.GRU(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)
        # how to define the input_size and hidden_size?
        # test num_layers param, what does it affect?
        self.gru = torch.nn.GRU(input_size=total_ch, # ????
                                hidden_size=512,    # the more hidden size, the complexer memory
                                num_layers=2, #2 or 3       # does this represent the number of GRU blocks? or say the number of consecutive frames we wanna consider?
                                bias=True, 
                                batch_first=True,    # True
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
        #print("Temporal_start", x_att.shape, h_state)
        # h_n itself should be an input for GRU, otherwise useless, dont forget the hidden state
        out, h_n = self.gru(x_att, h_state) # read the source, there are 2 outputs, but what is h_n here? should be the hidden state of the last layer?
        # mind the coherence of the input and output of the RNN layer 
        
        #reshape the output
        # for input frames, the temporal information should be considered, how to define the input_size and hidden_size?

        #print("Temporal out", out.shape)
        #print("Temporal h_n", h_n.shape)
        return out, h_n   # okay one important thing is to use h_n or out for fc layer?
        # answer: use out, since it contains the info of all "time steps" (or frame steps). However, h_n only contains the info of the last time step. 
        

        # set learning rate for diff layers




#########################################################################

# class FeatureExtraction(torch.nn.Module):
#     feature_extractor = ResNetFeatureExtractor()
#     feature_extractor.eval()  
#     transform = transforms.Compose([ # pre processing the images to match the resnet's training statistics
#         transforms.Resize((128, 128)), 
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalization of the pixel values
#     ])

#     def extract_features(self, image_path):
#         img = Image.open(image_path).convert("RGB") 
#         img = self.transform(img).unsqueeze(0) # apply the transformations, batch size is set to 1
#         # with torch.no_grad():
#         features = self.feature_extractor(img)  
        
#         return features

class FeatureExtraction(torch.nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        self.feature_extractor = ResNetFeatureExtractor()
        self.feature_extractor.eval()

    def extract_features(self, img_tensor):
        # img_tensor = img_tensor.unsqueeze(0)  # add batch dimension, no...
        with torch.no_grad():
            features = self.feature_extractor(img_tensor)
        return features



#########################################################################

class GazePrediction(nn.Module):
    """
    FC layer
    input shape: (seq_len, bs, ch)  # output of GRU 
    reshape input to ( bs, seq_len*ch)  # flatten the input
    output shape: (bs, num_classes)  # output of the model
    """
    def __init__(self, input_dim, num_classes):
        super(GazePrediction, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        #print("enter FC layer")
        # Ensure x is flat going into the FC layer (it should already be flat if coming from GAP)
        x = x.view(x.size(0), -1)  # Flatten to [bs, features]
        x = self.fc(x)
        #print("GazePrediction", x.shape)
        return x

class WholeModel(nn.Module): ## Sequence Length=batchsize !!
    def __init__(self):
        super(WholeModel, self).__init__()
        self.layers= (nn.ModuleList([
                FeatureExtraction(),
                FeatureFusion(),
                Attention(),
                Temporal(),
                GazePrediction(input_dim=512, num_classes=2)  # why sequence_length = 64? h*w*bs, yes indeed :)
                # RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x8192 and 32768x3) from 8192/512=16 I know seq_len = 16, but why?
                # answer: read the source code of GRU, the output of GRU is (seq_len, bs, hidden_size), so the input of FC layer should be (bs, seq_len*hidden_size)
            ]))
        
        # self.left_eye = self.layers[0]
        # self.right_eye = self.layers[0]
        # self.face = self.layers[0]

    def forward(self, left_eye_img, right_eye_img, face_img):
        # Extract features
        left_eye = self.layers[0].extract_features(left_eye_img)
        if (torch.isnan(left_eye).any()):
            print(f"left_eye feature NaN check: {torch.isnan(left_eye).any()}")

        right_eye = self.layers[0].extract_features(right_eye_img)
        if (torch.isnan(right_eye).any()):
            print(f"right_eye feature NaN check: {torch.isnan(right_eye).any()}")

        face = self.layers[0].extract_features(face_img)
        if (torch.isnan(face).any()):
            print(f"face feature NaN check: {torch.isnan(face).any()}")

        # Fusion
        fusioned_feature = self.layers[1](left_eye, right_eye, face)
        if (torch.isnan(fusioned_feature).any()):
            print(f"fusion NaN check: {torch.isnan(fusioned_feature).any()}")

        # Attention
        Attention_map = self.layers[2](fusioned_feature)
        if (torch.isnan(Attention_map).any()):
            print(f"attention NaN check: {torch.isnan(Attention_map).any()}")

        # Reshape for GRU
        # bs, seq_len, ch = Attention_map.shape
        # Attention_map = Attention_map.reshape(seq_len, bs, ch) # change to (ch, bs, seq_len), or bs first (bs, ch, seq_len)?
        # Attention_map = Attention_map.reshape(bs, ch, seq_len)

        # GRU
        gru_out, _ = self.layers[3](Attention_map) # batch_first=False, so the input should be (ch, bs, seq_len)
        if (torch.isnan(gru_out).any()):
            print(f"GRU output NaN check: {torch.isnan(gru_out).any()}")

        # GAP
        gap = torch.mean(gru_out, dim=0)
        if (torch.isnan(gap).any()):
            print(f"GAPP prediction NaN check: {torch.isnan(gap).any()}")

        # FC layer
        pred = self.layers[4](gap)
        if(torch.isnan(pred).any()):

            print(f"Final prediction NaN check: {torch.isnan(pred).any()}")

        return pred
    
    
