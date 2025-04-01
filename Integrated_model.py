## layer 1, understand strides and kernel_size concepts, we do not freeze the layers i think? maybe if there is overfitting we can freeze resnet for less training parameters
# TODO: Ask about copying codes from documentation
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
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
        img_tensor = img_tensor.unsqueeze(0)  # add batch dimension
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
        print("enter FC layer")
        # Ensure x is flat going into the FC layer (it should already be flat if coming from GAP)
        x = x.view(x.size(0), -1)  # Flatten to [bs, features]
        x = self.fc(x)
        print("GazePrediction", x.shape)
        return x

class WholeModel(nn.Module): ## Sequence Length=batchsize !!
    def __init__(self):
        super(WholeModel, self).__init__()
        self.layers= (nn.ModuleList([
                FeatureExtraction(),
                FeatureFusion(),
                Attention(),
                Temporal(),
                GazePrediction(input_dim=512, num_classes=3)  # why sequence_length = 64? h*w*bs, yes indeed :)
                # RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x8192 and 32768x3) from 8192/512=16 I know seq_len = 16, but why?
                # answer: read the source code of GRU, the output of GRU is (seq_len, bs, hidden_size), so the input of FC layer should be (bs, seq_len*hidden_size)
            ]))
        
        # self.left_eye = self.layers[0]
        # self.right_eye = self.layers[0]
        # self.face = self.layers[0]
        

    def forward(self, left_eye_img, right_eye_img, face_img):
        # calculate 3 input features in parallel
        left_eye = self.layers[0].extract_features(left_eye_img)
        right_eye = self.layers[0].extract_features(right_eye_img)
        face = self.layers[0].extract_features(face_img)
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
        # gru_out = gru_out.reshape(gru_out.shape[0], -1)   
        #print("gru_out", gru_out.shape)
        gap = torch.mean(gru_out, dim=0)
        print("asdfasdfasdfasdf")
        print(gap.shape)
        print("asdfasdfasdfasdf")
        pred = self.layers[4](gap)  # FC layer handles the rest
        print("WholeModel", pred.shape)
        return pred
    
    

###########################################################

# model = WholeModel()

# output = model("C:/Users/rohan/Desktop/Master/Master Thesis/Master-Thesis/Dataset-Test/Output Folder/webcam_r/left_eye/left_eye_0000.jpg", "C:/Users/rohan/Desktop/Master/Master Thesis/Master-Thesis/Dataset-Test/Output Folder/webcam_r/right_eye/right_eye_0000.jpg", "C:/Users/rohan/Desktop/Master/Master Thesis/Master-Thesis/Dataset-Test/Output Folder/webcam_r/face/face_0000.jpg")

#########################################################


# class GazeDatasetFromPaths(Dataset):
#     def __init__(self, folder_path, label_path):
#         self.folder_path = folder_path
#         self.labels = pd.read_csv(label_path, header=None).values.astype('float32') # converted to numpy array
#         self.left_eye_files = sorted(os.listdir(os.path.join(folder_path, "left_eye")))
#         self.right_eye_files = sorted(os.listdir(os.path.join(folder_path, "right_eye")))
#         self.face_files = sorted(os.listdir(os.path.join(folder_path, "face")))

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         left_path = os.path.join(self.folder_path, "left_eye", self.left_eye_files[idx])
#         right_path = os.path.join(self.folder_path, "right_eye", self.right_eye_files[idx])
#         face_path = os.path.join(self.folder_path, "face", self.face_files[idx])
#         label = torch.tensor(self.labels[idx], dtype=torch.float32)
#         print("--------------------------qwe-----------")
#         print(label.shape)
#         print("--------------------------qwe-----------")
#         return left_path, right_path, face_path, label

# def dot_product_loss(pred, target):

#     pred = nn.functional.normalize(pred, p=2, dim=1) # Resulting vector will have the correct direction but unit vector
#     target = nn.functional.normalize(target, p=2, dim=1)
#     print(pred)
#     print(target)
#     return torch.sum(pred * target, dim=1).mean()

# def angular_error(pred, target):
#     pred = nn.functional.normalize(pred, p=2, dim=1)
#     target = nn.functional.normalize(target, p=2, dim=1)
#     cos_sim = torch.sum(pred * target, dim=1)
#     return torch.acos(cos_sim) * (180.0 / torch.pi)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = WholeModel().to(device)

# dataset_path = "C:/Users/rohan/Desktop/Master/Master Thesis/Master-Thesis/Dataset-Test/Output Folder/webcam_r"
# label_excel = "C:/Users/rohan/Desktop/Master/Master Thesis/Master-Thesis/Dataset-Test/Output Folder/data.csv"

# dataset = GazeDatasetFromPaths(dataset_path, label_excel)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # shuffle true? yes, cause label is included so no issue

# optimizer = optim.Adam(model.parameters(), lr=0.001)

# epochs = 5
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0.0
#     total_ang_error = 0.0

#     for left_paths, right_paths, face_paths, labels in dataloader:
#         labels = labels.to(device) # move labels to same device
#         predictions = []

#         for i in range(len(left_paths)):
#             pred = model(left_paths[i], right_paths[i], face_paths[i]).squeeze(0)
#             predictions.append(pred)

#         predictions = torch.stack(predictions)
#         loss = dot_product_loss(predictions, labels)
#         ang_err = angular_error(predictions, labels).mean()

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         total_ang_error += ang_err.item()

#     print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Mean Angular Error: {total_ang_error/len(dataloader):.2f}°")
