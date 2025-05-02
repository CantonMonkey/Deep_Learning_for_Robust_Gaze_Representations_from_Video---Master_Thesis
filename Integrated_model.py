#no RELU, low dropout, and overfitting.

## layer 1, understand strides and kernel_size concepts, we do not freeze the layers i think? maybe if there is overfitting we can freeze resnet for less training parameters
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import logging
import sys

'''feature fusion'''
from fusion import AFF, iAFF, DAF, MS_CAM
from aff_resnet import resnet18 as aff_resnet18
''''''

import logging
logger = logging.getLogger(__name__)

# https://flypix.ai/blog/image-recognition-algorithms/ why I choosed only first few layers for feature extraction, upto middle layers are sufficient to provide information about eyes
# In higher layers of the network, detailed pixel information is lost whilethe high level content of the image is preserved. Clear Explanation is here(https://ai.stackexchange.com/questions/30038/why-do-we-lose-detail-of-an-image-as-we-go-deeper-into-a-convnet)
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:6])  # output shape: (bs, 64, h/4, w/4)  # 64 channels, 1/4 of the original image size

        for param in self.feature_extractor.parameters():
            param.requires_grad = False


    def forward(self, x):
        x = self.feature_extractor(x)  
        return x  # (bs, ch, h, w)




class FeatureFusion(torch.nn.Module):
    def __init__(self, total_ch=384):
        super(FeatureFusion, self).__init__()
        self.gn = torch.nn.GroupNorm(24, total_ch)
        self.channel_attention = MS_CAM(channels=total_ch)
        self.dropout = nn.Dropout(0.15)
            
    def forward(self, left_eye, right_eye, face):
        concate = torch.cat((left_eye, right_eye, face), 1)
        out = self.gn(concate)
        #out = self.dropout(out)
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
    def __init__(self, total_ch=384):    # the gaze_dims should be defined later, according to the dataset and what we wanna predict, for instance, 3 gaze dims, PoG (x,y,z)
        super(Attention, self).__init__()
        # self.self_att = Transformer(
        #         dim = total_ch,   # if not using the total_ch, should project the input to the dim of the transformer first
        #         depth = 3,
        #         heads = 8,
        #         dim_head = total_ch//8,  #dim//heads
        #         mlp_dim = 512,  # the hidden layer dim of the mlp (the hidden layer of the feedforward network, which is applied to each position (each token) separately and identically)
        #         dropout = 0.1
        #         # def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.)
        # )
        self.self_att = Transformer(
            dim = total_ch,
            depth = 2, # decrease the depth a bit 
            heads = 9,
            dim_head = 64,  # Fixed dimension may be more stable than relative dimension. right? maybe other things 160
            mlp_dim = 512,
            dropout = 0.15
        )


    def forward(self, out):
        bs, c, h, w = out.shape
        x_att = out.reshape(bs, c, h * w).transpose(1, 2)   # (bs, h*w, c) --- (bs, seq_len, features)
        x_att = self.self_att(x_att)  # output shape (bs, h*w, c)

        return x_att

     


class Temporal(torch.nn.Module):
    """
    Temporal Layer
    input shape: (bs, h*w, ch)    # ==(bs, seq_len, ch)
    output shape: (bs, seq_len, ch)  # seq_len = h*w 
    h_n shape: (num_layers, bs, hidden_size)
    """
    def __init__(self, total_ch = 384):    # the gaze_dims should be defined later, according to the dataset and what we wanna predict, for instance, 3 gaze dims, PoG (x,y,z)
        super(Temporal, self).__init__()
        self.gru = torch.nn.GRU(input_size=total_ch, # ????
                                hidden_size=512,    # the more hidden size, the complexer memory
                                num_layers=2, #2 or 3       # does this represent the number of GRU blocks? or say the number of consecutive frames we wanna consider?
                                bias=True, 
                                batch_first=True,    # True
                                dropout=0.15, 
                                bidirectional=False,   
                                device=None, 
                                dtype=None)
        
        # self.gru = nn.GRU(
        #     input_size=384,  # Channel number of attention output from base model
        #     hidden_size=512,
        #     num_layers=2,
        #     batch_first=True,
        #     dropout=0.1
        # )
        
    
        

    def forward(self, x_att, h_state=None):
        out, h_n = self.gru(x_att, h_state) # read the source, there are 2 outputs, but what is h_n here? should be the hidden state of the last layer?

        return out, h_n   # okay one important thing is to use h_n or out for fc layer?
        # answer: use out, since it contains the info of all "time steps" (or frame steps). However, h_n only contains the info of the last time step. 


class GazePrediction(nn.Module):
    """
    FC layer
    input shape: (seq_len, bs, ch)  # output of GRU 
    reshape input to ( bs, seq_len*ch)  # flatten the input
    output shape: (bs, num_classes)  # output of the model
    """
    def __init__(self, input_dim, num_classes):
        super(GazePrediction, self).__init__()
        # self.fc = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(0.15)
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        #print("enter FC layer")
        # Ensure x is flat going into the FC layer (it should already be flat if coming from GAP)
        x = x.view(x.size(0), -1)  # Flatten to [bs, features]
        #x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

## CNN + Attention
class WholeModel(nn.Module):
    def __init__(self):
        super(WholeModel, self).__init__()

        # Feature extractors
        self.left_eye_extractor = ResNetFeatureExtractor()
        self.right_eye_extractor = ResNetFeatureExtractor()
        self.face_extractor = ResNetFeatureExtractor()
        self.uni_extractor = ResNetFeatureExtractor()
        
        # Feature fusion
        self.feature_fusion = FeatureFusion()
        
        # Attention mechanism
        self.attention = Attention()
        
        # GRU and prediction layers removed
    
    def forward(self, left_eye_img, right_eye_img, face_img):
        # Feature extraction
        left_feat = self.uni_extractor(left_eye_img)
        right_feat = self.uni_extractor(right_eye_img)
        face_feat = self.uni_extractor(face_img)
        
        # Feature fusion
        fused = self.feature_fusion(left_feat, right_feat, face_feat)
        
        # Attention processing
        att_out = self.attention(fused)  # [B, H*W, C]
        
        # Return features directly without prediction
        return att_out

class SequentialWholeModel(nn.Module):
    def __init__(self, base_model=None):
        super(SequentialWholeModel, self).__init__()
        
        # Use existing model or create a new one
        if base_model is not None:
            self.base_model = base_model
        else:
            self.base_model = WholeModel()
            
        # Add sequential GRU layer
        # self.gru = nn.GRU(
        #     input_size=384,  # Channel number of attention output from base model
        #     hidden_size=512,
        #     num_layers=2,
        #     batch_first=True,
        #     dropout=0.1
        # )
        self.gru = Temporal(total_ch=384)  # Use the custom GRU class defined above
        
        # Prediction layer transplanted from WholeModel
        self.prediction = GazePrediction(input_dim=512, num_classes=2)
    
    def forward(self, left_eyes, right_eyes, faces):
        """
        Process sequence input
        Args:
            left_eyes: [B, seq_len, C, H, W]
            right_eyes: [B, seq_len, C, H, W]
            faces: [B, seq_len, C, H, W]
        Returns:
            outputs: [B, seq_len, 2]
        """
        seq_len = left_eyes.shape[1]
        batch_size = left_eyes.shape[0]
        
        # Prepare to store features for each frame
        frame_features = []
        
        # Process each frame in the sequence to get features
        for t in range(seq_len):
            # Get data for current time step
            left_t = left_eyes[:, t]   # [B, C, H, W]
            right_t = right_eyes[:, t] # [B, C, H, W]
            face_t = faces[:, t]       # [B, C, H, W]
            
            # Process current frame through base model to get features
            features_t = self.base_model(left_t, right_t, face_t)  # [B, H*W, C]
            
            # Global average pooling to reduce feature dimensions
            pooled_features = torch.mean(features_t, dim=1)  # [B, C]
            
            frame_features.append(pooled_features)
        
        # Stack features from all frames
        frame_features = torch.stack(frame_features, dim=1)  # [B, seq_len, C]
        
        # Process sequence through GRU
        rnn_out, _ = self.gru(frame_features)  # [B, seq_len, hidden_size]
        
        # Prepare to store predictions for each time step
        outputs = []
        
        # Make prediction for each time step
        for t in range(seq_len):
            time_feat = rnn_out[:, t]  # [B, hidden_size]
            pred_t = self.prediction(time_feat)  # [B, 2]
            outputs.append(pred_t)
        
        # Stack predictions from all time steps
        outputs = torch.stack(outputs, dim=1)  # [B, seq_len, 2]
        
        return outputs


    