
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import logging
# from vit_pytorch import ViT
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
from vit_pytorch.vit import Transformer
import logging
logger = logging.getLogger(__name__)

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
    def __init__(self, total_ch=288):
        super(FeatureFusion, self).__init__()
        self.gn = torch.nn.GroupNorm(24, total_ch)
            
    def forward(self, left_eye, right_eye, face):
        concate = torch.cat((left_eye, right_eye, face), 1)
        out = self.gn(concate)
        return out





class Attention(torch.nn.Module):
    """
    Attention Layer
    seq_len = h*w (number of patch = 1, hence token = h*w), features = ch
    class input shape: (bs, ch, h, w)
    in forward, reshape to (bs, h*w, ch)
    output shape: (bs, h*w, ch)
    """
    def __init__(self, total_ch=288):    # the gaze_dims should be defined later, according to the dataset and what we wanna predict, for instance, 3 gaze dims, PoG (x,y,z)
        super(Attention, self).__init__()
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
    def __init__(self, total_ch = 288):    # the gaze_dims should be defined later, according to the dataset and what we wanna predict, for instance, 3 gaze dims, PoG (x,y,z)
        super(Temporal, self).__init__()
        self.gru = torch.nn.GRU(input_size=total_ch,
                                hidden_size=512,    # the more hidden size, the complexer memory
                                num_layers=2, #2 or 3       # does this represent the number of GRU blocks? or say the number of consecutive frames we wanna consider?
                                bias=True, 
                                batch_first=True,    
                                dropout=0.15, 
                                bidirectional=False,   
                                device=None, 
                                dtype=None)
        
        
    
        

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
        self.dropout = nn.Dropout(0.2)
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

        self.uni_extractor = ResNetFeatureExtractor()

        self.face_ch_reduce = nn.Conv2d(128, 32, kernel_size=1)
        
        self.feature_fusion = FeatureFusion()
        
        self.attention = Attention()
    
    def forward(self, left_eye_img, right_eye_img, face_img):
        # Feature extraction
        left_feat = self.uni_extractor(left_eye_img)
        right_feat = self.uni_extractor(right_eye_img)
        face_feat = self.uni_extractor(face_img)
    
        # face feat reduction
        face_feat = self.face_ch_reduce(face_feat)
        
        # Feature fusion (GN)
        fused = self.feature_fusion(left_feat, right_feat, face_feat)
        
        # ViT
        att_out = self.attention(fused)  # [B, H*W, C]
        
        return att_out

class SequentialWholeModel(nn.Module):
    def __init__(self, base_model=None):
        super(SequentialWholeModel, self).__init__()
        
        # Use existing model or create a new one
        if base_model is not None:
            self.base_model = base_model
        else:
            self.base_model = WholeModel()
            

        self.gru = Temporal(total_ch=288)
        
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
        self.uni_extractor = ResNetFeatureExtractor()
        
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
