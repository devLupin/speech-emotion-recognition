"""
    - 패치 단위 학습
        - 2D CNN으로부터 추출된 값을 패치 단위로 쪼개고, 이를 트랜스포머에 입력
            - CNN은 공간 정보만 있기 때문
        - 정확도 향상 안됨.
    - cnn_transformer_lstm()에 DenseNet 개념인 concat 적용
        - 정확도 향상 안됨.
        - training acc만 빠르게 올라가고, val acc는 예전과 똑같이 올라감.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from einops import rearrange
import yaml

def load_yaml(load_path):
    """load yaml file"""
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)

    return loaded

cfg = load_yaml('./configs/config.yaml')


def arrange_for_patch(x, num_patches=16):
    h = x.shape[0]
    w = x.shape[1]
    c = w//num_patches

    x = x.view([h, -1, c])  # -1 = unknown dim
    return x


class parallel_all_you_want(nn.Module):
    # Define all layers present in the network
    def __init__(self, num_emotions):
        super().__init__()

        ################ TRANSFORMER BLOCK #############################
        # maxpool the input feature map/tensor to the transformer
        # a rectangular kernel worked better here for the rectangular input spectrogram feature map/tensor
        self.transformer_maxpool = nn.MaxPool2d(
            kernel_size=[1, 4], stride=[1, 4])

        # define single transformer encoder layer
        # self-attention + feedforward network from "Attention is All You Need" paper
        # 4 multi-head self-attention layers each with 40-->512--->40 feedforward network
        transformer_layer = nn.TransformerEncoderLayer(
            # input feature (frequency) dim after maxpooling 40*282 -> 40*70 (MFC*time)
            d_model=40,
            nhead=4,  # 4 self-attention layers in each multi-head self-attention layer in each encoder block
            # 2 linear layers in each encoder block's feedforward network: dim 40-->512--->40
            dim_feedforward=512,
            dropout=0.4,
            activation='relu'  # ReLU: avoid saturation/tame gradient/reduce compute time
        )

        # I'm using 4 instead of the 6 identical stacked encoder layrs used in Attention is All You Need paper
        # Complete transformer block contains 4 full transformer encoder layers (each w/ multihead self-attention+feedforward)
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, num_layers=4)

        ############### 1ST PARALLEL 2D CONVOLUTION BLOCK ############
        # 3 sequential conv2D layers: (1,40,282) --> (16, 20, 141) -> (32, 5, 35) -> (64, 1, 8)
        self.conv2Dblock1 = nn.Sequential(

            # 1st 2D convolution layer
            nn.Conv2d(
                in_channels=1,  # input volume depth == input channel dim == 1
                out_channels=16,  # expand output feature map volume's depth to 16
                kernel_size=3,  # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
            ),
            # batch normalize the output feature map before activation
            nn.BatchNorm2d(16),
            nn.ReLU(),  # feature map --> activation map
            # typical maxpool kernel size
            nn.MaxPool2d(kernel_size=2, stride=2),
            # randomly zero 30% of 1st layer's output feature map in training
            nn.Dropout(p=0.3),

            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16,
                out_channels=32,  # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # increase maxpool kernel for subsequent filters
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),

            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64,  # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        ############### 2ND PARALLEL 2D CONVOLUTION BLOCK ############
        # 3 sequential conv2D layers: (1,40,282) --> (16, 20, 141) -> (32, 5, 35) -> (64, 1, 8)
        self.conv2Dblock2 = nn.Sequential(

            # 1st 2D convolution layer
            nn.Conv2d(
                in_channels=1,  # input volume depth == input channel dim == 1
                out_channels=16,  # expand output feature map volume's depth to 16
                kernel_size=3,  # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
            ),
            # batch normalize the output feature map before activation
            nn.BatchNorm2d(16),
            nn.ReLU(),  # feature map --> activation map
            # typical maxpool kernel size
            nn.MaxPool2d(kernel_size=2, stride=2),
            # randomly zero 30% of 1st layer's output feature map in training
            nn.Dropout(p=0.3),

            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16,
                out_channels=32,  # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # increase maxpool kernel for subsequent filters
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),

            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64,  # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )

        ################# FINAL LINEAR BLOCK ####################
        # Linear softmax layer to take final concatenated embedding tensor
        #    from parallel 2D convolutional and transformer blocks, output 8 logits
        # Each full convolution block outputs (64*1*8) embedding flattened to dim 512 1D array
        # Full transformer block outputs 40*70 feature map, which we time-avg to dim 40 1D array
        # 512*2+40 == 1064 input features --> 8 output emotions
        self.fc1_linear = nn.Linear(512*2+40, num_emotions)

        ### Softmax layer for the 8 output logits from final FC linear layer
        self.softmax_out = nn.Softmax(dim=1)  # dim==1 is the freq embedding

    # define one complete parallel fwd pass of input feature tensor thru 2*conv+1*transformer blocks
    def forward(self, x):

        ############ 1st parallel Conv2D block: 4 Convolutional layers ############################
        # create final feature embedding from 1st convolutional layer
        # input features pased through 4 sequential 2D convolutional layers
        # x == N/batch * channel * freq * time
        conv2d_embedding1 = self.conv2Dblock1(x)

        # flatten final 64*1*8 feature map from convolutional layers to length 512 1D array
        # skip the 1st (N/batch) dimension when flattening
        conv2d_embedding1 = torch.flatten(conv2d_embedding1, start_dim=1)

        ############ 2nd parallel Conv2D block: 4 Convolutional layers #############################
        # create final feature embedding from 2nd convolutional layer
        # input features pased through 4 sequential 2D convolutional layers
        # x == N/batch * channel * freq * time
        conv2d_embedding2 = self.conv2Dblock2(x)

        # flatten final 64*1*8 feature map from convolutional layers to length 512 1D array
        # skip the 1st (N/batch) dimension when flattening
        conv2d_embedding2 = torch.flatten(conv2d_embedding2, start_dim=1)

        ########## 4-encoder-layer Transformer block w/ 40-->512-->40 feedfwd network ##############
        # maxpool input feature map: 1*40*282 w/ 1*4 kernel --> 1*40*70
        x_maxpool = self.transformer_maxpool(x)

        # remove channel dim: 1*40*70 --> 40*70
        x_maxpool_reduced = torch.squeeze(x_maxpool, 1)

        # convert maxpooled feature map format: batch * freq * time ---> time * batch * freq format
        # because transformer encoder layer requires tensor in format: time * batch * embedding (freq)
        x = x_maxpool_reduced.permute(2, 0, 1)

        # finally, pass reduced input feature map x into transformer encoder layers
        transformer_output = self.transformer_encoder(x)

        # create final feature emedding from transformer layer by taking mean in the time dimension (now the 0th dim)
        # transformer outputs 2x40 (MFCC embedding*time) feature map, take mean of columns i.e. take time average
        transformer_embedding = torch.mean(
            transformer_output, dim=0)  # dim 40x70 --> 40

        ############# concatenate freq embeddings from convolutional and transformer blocks ######
        # concatenate embedding tensors output by parallel 2*conv and 1*transformer blocks
        complete_embedding = torch.cat(
            [conv2d_embedding1, conv2d_embedding2, transformer_embedding], dim=1)

        ######### final FC linear layer, need logits for loss #########################
        output_logits = self.fc1_linear(complete_embedding)

        ######### Final Softmax layer: use logits from FC linear, get softmax for prediction ######
        output_softmax = self.softmax_out(output_logits)

        # need output logits to compute cross entropy loss, need softmax probabilities to predict class
        return output_logits, output_softmax


class cnn_to_transformer(nn.Module):
    def __init__(self, num_emotions) -> None:
        super().__init__()

        self.conv2Dblock = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )

        self.transformer_maxpool = nn.MaxPool2d(
            kernel_size=[1, 4], stride=[1, 4])

        transformer_layer = nn.TransformerEncoderLayer(
            # input feature (frequency) dim after maxpooling 40*282 -> 40*70 (MFC*time)
            d_model=32,
            nhead=4,  # 4 self-attention layers in each multi-head self-attention layer in each encoder block
            # 2 linear layers in each encoder block's feedforward network: dim 40-->512--->40
            dim_feedforward=512,
            dropout=0.4,
            activation='relu'  # ReLU: avoid saturation/tame gradient/reduce compute time
        )

        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, num_layers=4)

        self.fc_512 = nn.Linear(32, 512)

        self.fc_emotion = nn.Linear(512, 8)

        self.softmax_out = nn.Softmax(dim=1)

    def forward(self, x):
        conv2d_embd = self.conv2Dblock(x)
        conv2d_embd = torch.squeeze(conv2d_embd, 2)
        conv2d_embedding = torch.flatten(conv2d_embd, start_dim=1)

        patch = arrange_for_patch(conv2d_embedding)

        # batch * time * freq -> time * batch * freq
        patch = patch.permute(1, 0, 2)

        transformer_out = self.transformer_encoder(patch)
        transformer_embd = torch.mean(transformer_out, dim=0)

        fc_1024_output = self.fc_512(transformer_embd)
        output_logits = self.fc_emotion(fc_1024_output)
        output_softmax = self.softmax_out(output_logits)

        return output_logits, output_softmax


class cnn_lstm_transformer(nn.Module):
    def __init__(self, num_emotions) -> None:
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=[1, 4], stride=[1, 4])

        self.lstm = nn.LSTM(input_size=40, hidden_size=128, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.4)

        transformer_layer = nn.TransformerEncoderLayer(
            # input feature (frequency) dim after maxpooling 40*282 -> 40*70 (MFC*time)
            d_model=40,
            nhead=4,  # 4 self-attention layers in each multi-head self-attention layer in each encoder block
            # 2 linear layers in each encoder block's feedforward network: dim 40-->512--->40
            dim_feedforward=512,
            dropout=0.4,
            activation='relu'  # ReLU: avoid saturation/tame gradient/reduce compute time
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, num_layers=4)

        self.conv2Dblock = nn.Sequential(

            # 1st 2D convolution layer
            nn.Conv2d(
                in_channels=1,  # input volume depth == input channel dim == 1
                out_channels=16,  # expand output feature map volume's depth to 16
                kernel_size=3,  # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
            ),
            # batch normalize the output feature map before activation
            nn.BatchNorm2d(16),
            nn.ReLU(),  # feature map --> activation map
            # typical maxpool kernel size
            nn.MaxPool2d(kernel_size=2, stride=2),
            # randomly zero 30% of 1st layer's output feature map in training
            nn.Dropout(p=0.3),

            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16,
                out_channels=32,  # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # increase maxpool kernel for subsequent filters
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),

            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64,  # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )

        self.fc_linear = nn.Linear(512+256+40, num_emotions)
        self.softmax_out = nn.Softmax(dim=1)

    def forward(self, x):
        conv2d_embedding = self.conv2Dblock(x)
        conv2d_embedding = torch.flatten(conv2d_embedding, start_dim=1)

        x_reduced = self.maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(0, 2, 1)      # (b, t, freq)

        lstm_embedding, (h, c) = self.lstm(x_reduced)
        lstm_embedding = self.dropout_lstm(lstm_embedding)
        lstm_embedding = torch.mean(lstm_embedding, dim=1)

        x_reduced = self.maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(2, 0, 1)

        transformer_output = self.transformer_encoder(x_reduced)
        transformer_embedding = torch.mean(transformer_output, dim=0)

        complete_embedding = torch.cat(
            [conv2d_embedding, lstm_embedding, transformer_embedding], dim=1)

        output_logits = self.fc_linear(complete_embedding)
        output_softmax = self.softmax_out(output_logits)

        return output_logits, output_softmax


class gru(nn.Module):
    def __init__(self, num_emotions) -> None:
        super().__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=[1, 4], stride=[1, 4])

        self.gru = nn.GRU(input_size=40, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout_gru = nn.Dropout(0.4)
        
        self.fc_linear = nn.Linear(256, num_emotions)
        self.softmax_out = nn.Softmax(dim=1)
        
    
    def forward(self, x):
        x_reduced = self.maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(0, 2, 1)
        
        gru_embedding, (h, c) = self.gru(x_reduced)
        gru_embedding = self.dropout_gru(gru_embedding)
        gru_embedding = torch.mean(gru_embedding, dim=1)
        
        output_logits = self.fc_linear(gru_embedding)
        output_softmax = self.softmax_out(output_logits)
        
        return output_logits, output_softmax
    

class cnn_transformer_lstm(nn.Module):
    def __init__(self, num_emotions) -> None:
        super().__init__()
        
        self.maxpool = nn.MaxPool2d(
            kernel_size=[1, 4], stride=[1, 4])

        transformer_layer = nn.TransformerEncoderLayer(
            # input feature (frequency) dim after maxpooling 40*282 -> 40*70 (MFC*time)
            d_model=40,
            nhead=4,  # 4 self-attention layers in each multi-head self-attention layer in each encoder block
            # 2 linear layers in each encoder block's feedforward network: dim 40-->512--->40
            dim_feedforward=512,
            dropout=0.4,
            activation='relu'  # ReLU: avoid saturation/tame gradient/reduce compute time
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, num_layers=4)
        
        self.lstm = nn.LSTM(input_size=40, hidden_size=128, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.4)

        self.conv2Dblock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input volume depth == input channel dim == 1
                out_channels=16,  # expand output feature map volume's depth to 16
                kernel_size=3,  # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
            ),
            # batch normalize the output feature map before activation
            nn.BatchNorm2d(16),
            nn.ReLU(),  # feature map --> activation map
            # typical maxpool kernel size
            nn.MaxPool2d(kernel_size=2, stride=2),
            # randomly zero 30% of 1st layer's output feature map in training
            nn.Dropout(p=0.3),

            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16,
                out_channels=32,  # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # increase maxpool kernel for subsequent filters
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),

            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64,  # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        
        self.conv2Dblock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input volume depth == input channel dim == 1
                out_channels=16,  # expand output feature map volume's depth to 16
                kernel_size=3,  # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
            ),
            # batch normalize the output feature map before activation
            nn.BatchNorm2d(16),
            nn.ReLU(),  # feature map --> activation map
            # typical maxpool kernel size
            nn.MaxPool2d(kernel_size=2, stride=2),
            # randomly zero 30% of 1st layer's output feature map in training
            nn.Dropout(p=0.3),

            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16,
                out_channels=32,  # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # increase maxpool kernel for subsequent filters
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),

            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64,  # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        
        self.conv2Dblock3 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input volume depth == input channel dim == 1
                out_channels=16,  # expand output feature map volume's depth to 16
                kernel_size=3,  # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
            ),
            # batch normalize the output feature map before activation
            nn.BatchNorm2d(16),
            nn.ReLU(),  # feature map --> activation map
            # typical maxpool kernel size
            nn.MaxPool2d(kernel_size=2, stride=2),
            # randomly zero 30% of 1st layer's output feature map in training
            nn.Dropout(p=0.3),

            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16,
                out_channels=32,  # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # increase maxpool kernel for subsequent filters
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),

            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64,  # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        
        self.fc_linear = nn.Linear(1832, num_emotions)
        self.softmax_out = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        conv2d_embedding1 = self.conv2Dblock1(x)
        conv2d_embedding1 = torch.flatten(conv2d_embedding1, start_dim=1)

        conv2d_embedding2 = self.conv2Dblock2(x)
        conv2d_embedding2 = torch.flatten(conv2d_embedding2, start_dim=1)

        conv2d_embedding3 = self.conv2Dblock3(x)
        conv2d_embedding3 = torch.flatten(conv2d_embedding3, start_dim=1)

        x_maxpool = self.maxpool(x)

        # remove channel dim: 1*40*70 --> 40*70
        x_maxpool_reduced = torch.squeeze(x_maxpool, 1)
        x = x_maxpool_reduced.permute(2, 0, 1)
        transformer_output = self.transformer_encoder(x)
        transformer_embedding = torch.mean(transformer_output, dim=0)  # dim 40x70 --> 40

        x_reduced = torch.squeeze(x_maxpool, 1)
        x_reduced = x_reduced.permute(0, 2, 1)      # (b, t, freq)

        lstm_embedding, (h, c) = self.lstm(x_reduced)
        lstm_embedding = self.dropout_lstm(lstm_embedding)
        lstm_embedding = torch.mean(lstm_embedding, dim=1)

        complete_embedding = torch.cat(
            [conv2d_embedding1, conv2d_embedding2, conv2d_embedding3,
             lstm_embedding, transformer_embedding], dim=1)

        output_logits = self.fc_linear(complete_embedding)

        output_softmax = self.softmax_out(output_logits)

        return output_logits, output_softmax
    

class two_gru_transformer(nn.Module):
    def __init__(self, num_emotions) -> None:
        super().__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=[1, 4], stride=[1, 4])

        self.gru1 = nn.GRU(input_size=40, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(input_size=40, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout_gru = nn.Dropout(0.4)
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.4,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4)
        
        self.conv2Dblock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input volume depth == input channel dim == 1
                out_channels=16,  # expand output feature map volume's depth to 16
                kernel_size=3,  # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
            ),
            # batch normalize the output feature map before activation
            nn.BatchNorm2d(16),
            nn.ReLU(),  # feature map --> activation map
            # typical maxpool kernel size
            nn.MaxPool2d(kernel_size=2, stride=2),
            # randomly zero 30% of 1st layer's output feature map in training
            nn.Dropout(p=0.3),

            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16,
                out_channels=32,  # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # increase maxpool kernel for subsequent filters
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),

            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64,  # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        
        self.conv2Dblock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input volume depth == input channel dim == 1
                out_channels=16,  # expand output feature map volume's depth to 16
                kernel_size=3,  # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
            ),
            # batch normalize the output feature map before activation
            nn.BatchNorm2d(16),
            nn.ReLU(),  # feature map --> activation map
            # typical maxpool kernel size
            nn.MaxPool2d(kernel_size=2, stride=2),
            # randomly zero 30% of 1st layer's output feature map in training
            nn.Dropout(p=0.3),

            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16,
                out_channels=32,  # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # increase maxpool kernel for subsequent filters
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),

            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64,  # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        
        self.conv2Dblock3 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input volume depth == input channel dim == 1
                out_channels=16,  # expand output feature map volume's depth to 16
                kernel_size=3,  # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
            ),
            # batch normalize the output feature map before activation
            nn.BatchNorm2d(16),
            nn.ReLU(),  # feature map --> activation map
            # typical maxpool kernel size
            nn.MaxPool2d(kernel_size=2, stride=2),
            # randomly zero 30% of 1st layer's output feature map in training
            nn.Dropout(p=0.3),

            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16,
                out_channels=32,  # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # increase maxpool kernel for subsequent filters
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),

            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64,  # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        
        self.fc_linear1 = nn.Linear(2048, 1024)
        self.fc_linear2 = nn.Linear(1024, 512)
        self.fc_linear3 = nn.Linear(512, num_emotions)
        self.softmax_out = nn.Softmax(dim=1)
        
    def forward(self, x):
        conv2d_embedding1 = self.conv2Dblock1(x)
        conv2d_embedding1 = torch.flatten(conv2d_embedding1, start_dim=1)

        conv2d_embedding2 = self.conv2Dblock2(x)
        conv2d_embedding2 = torch.flatten(conv2d_embedding2, start_dim=1)

        conv2d_embedding3 = self.conv2Dblock3(x)
        conv2d_embedding3 = torch.flatten(conv2d_embedding3, start_dim=1)
        
        x_reduced = self.maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(0, 2, 1)
        
        gru_embedding1, (h, c) = self.gru1(x_reduced)
        gru_embedding1 = self.dropout_gru(gru_embedding1)
        gru_embedding1 = torch.mean(gru_embedding1, dim=1)
        
        gru_embedding2, (h, c) = self.gru2(x_reduced)
        gru_embedding2 = self.dropout_gru(gru_embedding2)
        gru_embedding2 = torch.mean(gru_embedding2, dim=1)
        
        transformer_input = torch.cat([gru_embedding1, gru_embedding2], dim=1)
        transformer_embd = self.transformer_encoder(transformer_input)
    
        complete_embedding = torch.cat([conv2d_embedding1, conv2d_embedding2, conv2d_embedding3, transformer_embd], dim=1)
    
        logits = self.fc_linear1(complete_embedding)
        logits = self.fc_linear2(logits)
        output_logits = self.fc_linear3(logits)
    
        output_softmax = self.softmax_out(output_logits)
        
        return output_logits, output_softmax


class cnn_lstm_transformer_concat(nn.Module):
    def __init__(self, num_emotions) -> None:
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=[1, 4], stride=[1, 4])

        self.lstm = nn.LSTM(input_size=40, hidden_size=128, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.4)

        transformer_layer = nn.TransformerEncoderLayer(
            # input feature (frequency) dim after maxpooling 40*282 -> 40*70 (MFC*time)
            d_model=40,
            nhead=4,  # 4 self-attention layers in each multi-head self-attention layer in each encoder block
            # 2 linear layers in each encoder block's feedforward network: dim 40-->512--->40
            dim_feedforward=512,
            dropout=0.4,
            activation='relu'  # ReLU: avoid saturation/tame gradient/reduce compute time
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, num_layers=4)

        self.conv2Dblock = nn.Sequential(

            # 1st 2D convolution layer
            nn.Conv2d(
                in_channels=1,  # input volume depth == input channel dim == 1
                out_channels=16,  # expand output feature map volume's depth to 16
                kernel_size=3,  # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
            ),
            # batch normalize the output feature map before activation
            nn.BatchNorm2d(16),
            nn.ReLU(),  # feature map --> activation map
            # typical maxpool kernel size
            nn.MaxPool2d(kernel_size=2, stride=2),
            # randomly zero 30% of 1st layer's output feature map in training
            nn.Dropout(p=0.3),

            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16,
                out_channels=32,  # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # increase maxpool kernel for subsequent filters
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),

            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64,  # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )

        self.fc_linear1 = nn.Linear(12088, 2048)
        self.fc_linear2 = nn.Linear(2048, 1024)
        self.fc_linear3 = nn.Linear(1024, 512)
        self.fc_linear4 = nn.Linear(512, num_emotions)
        self.softmax_out = nn.Softmax(dim=1)

    def forward(self, x):
        identity = x
        identity = torch.squeeze(identity, 1)
        identity = torch.flatten(identity, start_dim=1)
        
        conv2d_embedding = self.conv2Dblock(x)
        conv2d_embedding = torch.flatten(conv2d_embedding, start_dim=1)     # 256, 512
        # conv2d_embedding = torch.cat([identity, conv2d_embedding], dim=1)

        x_reduced = self.maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(0, 2, 1)      # (b, t, freq)

        lstm_embedding, (h, c) = self.lstm(x_reduced)
        lstm_embedding = self.dropout_lstm(lstm_embedding)
        lstm_embedding = torch.mean(lstm_embedding, dim=1)
        # lstm_embedding = torch.cat([identity, lstm_embedding], dim=1)

        x_reduced = self.maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(2, 0, 1)

        transformer_output = self.transformer_encoder(x_reduced)
        transformer_embedding = torch.mean(transformer_output, dim=0)
        # transformer_embedding = torch.cat([identity, transformer_embedding], dim=1)

        complete_embedding = torch.cat(
            [identity, conv2d_embedding, lstm_embedding, transformer_embedding], dim=1)

        logits = self.fc_linear1(complete_embedding)
        logits = self.fc_linear2(logits)
        logits = self.fc_linear3(logits)
        output_logits = self.fc_linear4(logits)
        output_softmax = self.softmax_out(output_logits)

        return output_logits, output_softmax


class gru_lstm_transformer(nn.Module):
    def __init__(self, num_emotions) -> None:
        super().__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=[1, 4], stride=[1, 4])

        self.gru = nn.GRU(input_size=40, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout_gru = nn.Dropout(0.4)
        
        self.lstm = nn.LSTM(input_size=40, hidden_size=128, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.4)

        transformer_layer = nn.TransformerEncoderLayer(
            # input feature (frequency) dim after maxpooling 40*282 -> 40*70 (MFC*time)
            d_model=40,
            nhead=4,  # 4 self-attention layers in each multi-head self-attention layer in each encoder block
            # 2 linear layers in each encoder block's feedforward network: dim 40-->512--->40
            dim_feedforward=512,
            dropout=0.4,
            activation='relu'  # ReLU: avoid saturation/tame gradient/reduce compute time
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, num_layers=4)
        
        self.conv2Dblock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input volume depth == input channel dim == 1
                out_channels=16,  # expand output feature map volume's depth to 16
                kernel_size=3,  # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
            ),
            # batch normalize the output feature map before activation
            nn.BatchNorm2d(16),
            nn.ReLU(),  # feature map --> activation map
            # typical maxpool kernel size
            nn.MaxPool2d(kernel_size=2, stride=2),
            # randomly zero 30% of 1st layer's output feature map in training
            nn.Dropout(p=0.3),

            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16,
                out_channels=32,  # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # increase maxpool kernel for subsequent filters
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),

            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64,  # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        
        self.conv2Dblock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input volume depth == input channel dim == 1
                out_channels=16,  # expand output feature map volume's depth to 16
                kernel_size=3,  # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
            ),
            # batch normalize the output feature map before activation
            nn.BatchNorm2d(16),
            nn.ReLU(),  # feature map --> activation map
            # typical maxpool kernel size
            nn.MaxPool2d(kernel_size=2, stride=2),
            # randomly zero 30% of 1st layer's output feature map in training
            nn.Dropout(p=0.3),

            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16,
                out_channels=32,  # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # increase maxpool kernel for subsequent filters
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),

            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64,  # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        
        self.conv2Dblock3 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input volume depth == input channel dim == 1
                out_channels=16,  # expand output feature map volume's depth to 16
                kernel_size=3,  # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
            ),
            # batch normalize the output feature map before activation
            nn.BatchNorm2d(16),
            nn.ReLU(),  # feature map --> activation map
            # typical maxpool kernel size
            nn.MaxPool2d(kernel_size=2, stride=2),
            # randomly zero 30% of 1st layer's output feature map in training
            nn.Dropout(p=0.3),

            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16,
                out_channels=32,  # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # increase maxpool kernel for subsequent filters
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),

            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64,  # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        
        self.fc_linear1 = nn.Linear(2088, 1024)
        self.fc_linear2 = nn.Linear(1024, 512)
        self.fc_linear3 = nn.Linear(512, 256) 
        self.fc_linear4 = nn.Linear(256, num_emotions)
        self.softmax_out = nn.Softmax(dim=1)
        
    
    def forward(self, x):
        conv2d_embedding1 = self.conv2Dblock1(x)
        conv2d_embedding1 = torch.flatten(conv2d_embedding1, start_dim=1)

        conv2d_embedding2 = self.conv2Dblock2(x)
        conv2d_embedding2 = torch.flatten(conv2d_embedding2, start_dim=1)

        conv2d_embedding3 = self.conv2Dblock3(x)
        conv2d_embedding3 = torch.flatten(conv2d_embedding3, start_dim=1)
        
        x_reduced = self.maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(0, 2, 1)
        
        gru_embedding, (h, c) = self.gru(x_reduced)
        gru_embedding = self.dropout_gru(gru_embedding)
        gru_embedding = torch.mean(gru_embedding, dim=1)

        lstm_embedding, (h, c) = self.lstm(x_reduced)
        lstm_embedding = self.dropout_lstm(lstm_embedding)
        lstm_embedding = torch.mean(lstm_embedding, dim=1)
        
        x_reduced = self.maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(2, 0, 1)

        transformer_output = self.transformer_encoder(x_reduced)
        transformer_embedding = torch.mean(transformer_output, dim=0)
        
        complete_embedding = torch.cat([conv2d_embedding1, conv2d_embedding2, conv2d_embedding3, gru_embedding, lstm_embedding, transformer_embedding], dim=1)
        
        logits = self.fc_linear1(complete_embedding)
        logits = self.fc_linear2(logits)
        logits = self.fc_linear3(logits)
        output_logits = self.fc_linear4(logits)
        output_softmax = self.softmax_out(output_logits)
        
        return output_logits, output_softmax
    

# layer norm
class gru_lstm_transformer_ln(nn.Module):
    def __init__(self, num_emotions) -> None:
        super().__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=[1, 4], stride=[1, 4])

        self.relu = nn.ReLU()

        self.gru = nn.GRU(input_size=40, hidden_size=512, num_layers=4, batch_first=True, bidirectional=True, dropout=0.2)
        self.gru_ln = nn.LayerNorm(normalized_shape=1024, eps=1e-08)
        
        self.lstm = nn.LSTM(input_size=40, hidden_size=512, num_layers=4, batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm_ln = nn.LayerNorm(normalized_shape=1024, eps=1e-08)

        transformer_layer = nn.TransformerEncoderLayer(
            # input feature (frequency) dim after maxpooling 40*282 -> 40*70 (MFC*time)
            d_model=40,
            nhead=4,  # 4 self-attention layers in each multi-head self-attention layer in each encoder block
            # 2 linear layers in each encoder block's feedforward network: dim 40-->512--->40
            dim_feedforward=512,
            dropout=0.1,
            activation='relu'  # ReLU: avoid saturation/tame gradient/reduce compute time
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, num_layers=4)
        self.transformer_ln = nn.LayerNorm(normalized_shape=40, eps=1e-08)
        
        self.conv2Dblock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input volume depth == input channel dim == 1
                out_channels=16,  # expand output feature map volume's depth to 16
                kernel_size=3,  # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
            ),
            # batch normalize the output feature map before activation
            nn.BatchNorm2d(16),
            nn.ReLU(),  # feature map --> activation map
            # typical maxpool kernel size
            nn.MaxPool2d(kernel_size=2, stride=2),
            # randomly zero 30% of 1st layer's output feature map in training
            nn.Dropout(p=0.1),

            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16,
                out_channels=32,  # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # increase maxpool kernel for subsequent filters
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.1),

            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64,  # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.1),
        )
        
        self.conv2Dblock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input volume depth == input channel dim == 1
                out_channels=16,  # expand output feature map volume's depth to 16
                kernel_size=3,  # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
            ),
            # batch normalize the output feature map before activation
            nn.BatchNorm2d(16),
            nn.ReLU(),  # feature map --> activation map
            # typical maxpool kernel size
            nn.MaxPool2d(kernel_size=2, stride=2),
            # randomly zero 30% of 1st layer's output feature map in training
            nn.Dropout(p=0.1),

            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16,
                out_channels=32,  # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # increase maxpool kernel for subsequent filters
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.1),

            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64,  # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.1),
        )
        
        self.conv2Dblock3 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input volume depth == input channel dim == 1
                out_channels=16,  # expand output feature map volume's depth to 16
                kernel_size=3,  # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
            ),
            # batch normalize the output feature map before activation
            nn.BatchNorm2d(16),
            nn.ReLU(),  # feature map --> activation map
            # typical maxpool kernel size
            nn.MaxPool2d(kernel_size=2, stride=2),
            # randomly zero 30% of 1st layer's output feature map in training
            nn.Dropout(p=0.1),

            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16,
                out_channels=32,  # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # increase maxpool kernel for subsequent filters
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.1),

            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64,  # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.1),
        )
                
        self.fc_linear1 = nn.Linear(3624, 1024)
        self.fc_linear2 = nn.Linear(1024, 512)
        self.fc_linear3 = nn.Linear(512, 256) 
        self.fc_linear4 = nn.Linear(256, num_emotions)
        self.softmax_out = nn.Softmax(dim=1)
        
    
    def forward(self, x):
        print(x.shape)
        conv2d_embedding1 = self.conv2Dblock1(x)
        conv2d_embedding1 = torch.flatten(conv2d_embedding1, start_dim=1)

        conv2d_embedding2 = self.conv2Dblock2(x)
        conv2d_embedding2 = torch.flatten(conv2d_embedding2, start_dim=1)

        conv2d_embedding3 = self.conv2Dblock3(x)
        conv2d_embedding3 = torch.flatten(conv2d_embedding3, start_dim=1)
        
        x_reduced = self.maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(0, 2, 1)
        
        
        gru_embedding, h = self.gru(x_reduced)
        gru_embedding = torch.mean(gru_embedding, dim=1)
        gru_embedding = self.gru_ln(gru_embedding)
        gru_embedding = self.relu(gru_embedding)
        

        lstm_embedding, (h, c) = self.lstm(x_reduced)
        lstm_embedding = torch.mean(lstm_embedding, dim=1)
        lstm_embedding = self.lstm_ln(lstm_embedding)
        lstm_embedding = self.relu(lstm_embedding)
        
        x_reduced = self.maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(2, 0, 1)

        transformer_output = self.transformer_encoder(x_reduced)
        transformer_embedding = torch.mean(transformer_output, dim=0)
        transformer_embedding = self.transformer_ln(transformer_embedding)
        transformer_embedding = self.relu(transformer_embedding)
        
        complete_embedding = torch.cat([conv2d_embedding1, conv2d_embedding2, conv2d_embedding3, gru_embedding, lstm_embedding, transformer_embedding], dim=1)
        
        logits = self.fc_linear1(complete_embedding)
        logits = self.fc_linear2(logits)
        logits = self.fc_linear3(logits)
        output_logits = self.fc_linear4(logits)
        output_softmax = self.softmax_out(output_logits)
        
        return output_logits, output_softmax


class gru_lstm_transformer_transfer_resnet(nn.Module):
    def __init__(self, num_emotions) -> None:
        super().__init__()
        
        self.transform = transforms.Resize([224,224])
        self.resnet_patch_size = 40
        self.resnet_num_patches = 7
        
        self.model_ft1 = models.resnet18(pretrained=True)
        self.model_ft1 = torch.nn.Sequential(*(list(self.model_ft1.children())[:-1]))
        self.model_ft2 = models.resnet18(pretrained=True)
        self.model_ft2 = torch.nn.Sequential(*(list(self.model_ft2.children())[:-1]))
        self.model_ft3 = models.resnet18(pretrained=True)
        self.model_ft3 = torch.nn.Sequential(*(list(self.model_ft3.children())[:-1]))
        self.model_ft4 = models.resnet18(pretrained=True)
        self.model_ft4 = torch.nn.Sequential(*(list(self.model_ft4.children())[:-1]))
        self.model_ft5 = models.resnet18(pretrained=True)
        self.model_ft5 = torch.nn.Sequential(*(list(self.model_ft5.children())[:-1]))
        self.model_ft6 = models.resnet18(pretrained=True)
        self.model_ft6 = torch.nn.Sequential(*(list(self.model_ft6.children())[:-1]))
        self.model_ft7 = models.resnet18(pretrained=True)
        self.model_ft7 = torch.nn.Sequential(*(list(self.model_ft7.children())[:-1]))
        
        self.maxpool = nn.MaxPool2d(kernel_size=[1, 4], stride=[1, 4])

        self.relu = nn.ReLU()

        self.gru = nn.GRU(input_size=40, hidden_size=512, num_layers=4, batch_first=True, bidirectional=True, dropout=0.2)
        self.gru_ln = nn.LayerNorm(normalized_shape=1024, eps=1e-08)
        
        self.lstm = nn.LSTM(input_size=40, hidden_size=512, num_layers=4, batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm_ln = nn.LayerNorm(normalized_shape=1024, eps=1e-08)

        transformer_layer = nn.TransformerEncoderLayer(
            # input feature (frequency) dim after maxpooling 40*282 -> 40*70 (MFC*time)
            d_model=40,
            nhead=4,  # 4 self-attention layers in each multi-head self-attention layer in each encoder block
            # 2 linear layers in each encoder block's feedforward network: dim 40-->512--->40
            dim_feedforward=512,
            dropout=0.1,
            activation='relu'  # ReLU: avoid saturation/tame gradient/reduce compute time
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, num_layers=4)
        self.transformer_ln = nn.LayerNorm(normalized_shape=40, eps=1e-08)
        
        self.fc_linear1 = nn.Linear(4648, 1024)
        self.fc_linear2 = nn.Linear(1024, 512)
        self.fc_linear3 = nn.Linear(512, 256) 
        self.fc_linear4 = nn.Linear(256, num_emotions)
        self.softmax_out = nn.Softmax(dim=1)
        
        
    
    def forward(self, x):
        ft_input = rearrange(x, 'b c t f -> b t f c')   # (256, 7, 284, 1)
        ft_input = ft_input[:,:,:280,:]     # (256, 7, 280, 1)
        ft_input = rearrange(ft_input, 'b t (p p_f) c -> b p_f c t p', p=self.resnet_patch_size)     # (256, 7, 1, 40, 40)
        resize_ft_input = self.resize(ft_input)     # (256, 7, 1, 224, 224)
        resize_ft_input = torch.cat([resize_ft_input, resize_ft_input, resize_ft_input], dim=2)
        
        ft_output1 = self.model_ft1(resize_ft_input[:,0,:,:,:])
        ft_output1 = torch.flatten(ft_output1, start_dim=1)
        ft_output2 = self.model_ft2(resize_ft_input[:,1,:,:,:])
        ft_output2 = torch.flatten(ft_output2, start_dim=1)
        ft_output3 = self.model_ft3(resize_ft_input[:,2,:,:,:])
        ft_output3 = torch.flatten(ft_output3, start_dim=1)
        ft_output4 = self.model_ft4(resize_ft_input[:,3,:,:,:])
        ft_output4 = torch.flatten(ft_output4, start_dim=1)
        ft_output5 = self.model_ft5(resize_ft_input[:,4,:,:,:])
        ft_output5 = torch.flatten(ft_output5, start_dim=1)
        ft_output6 = self.model_ft6(resize_ft_input[:,5,:,:,:])
        ft_output6 = torch.flatten(ft_output6, start_dim=1)
        ft_output7 = self.model_ft7(resize_ft_input[:,6,:,:,:])
        ft_output7 = torch.flatten(ft_output7, start_dim=1)
        
        ft_embedding = torch.cat([ft_output1, ft_output2, ft_output3, ft_output4, ft_output5, ft_output6, ft_output7], dim=1)
        
        x_reduced = self.maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(0, 2, 1)
        
        
        gru_embedding, h = self.gru(x_reduced)
        gru_embedding = torch.mean(gru_embedding, dim=1)
        gru_embedding = self.gru_ln(gru_embedding)
        gru_embedding = self.relu(gru_embedding)
        

        lstm_embedding, (h, c) = self.lstm(x_reduced)
        lstm_embedding = torch.mean(lstm_embedding, dim=1)
        lstm_embedding = self.lstm_ln(lstm_embedding)
        lstm_embedding = self.relu(lstm_embedding)
        
        x_reduced = self.maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(2, 0, 1)

        transformer_output = self.transformer_encoder(x_reduced)
        transformer_embedding = torch.mean(transformer_output, dim=0)
        transformer_embedding = self.transformer_ln(transformer_embedding)
        transformer_embedding = self.relu(transformer_embedding)
        
        complete_embedding = torch.cat([ft_embedding, gru_embedding, transformer_embedding], dim=1)
        
        logits = self.fc_linear1(complete_embedding)
        logits = self.fc_linear2(logits)
        logits = self.fc_linear3(logits)
        output_logits = self.fc_linear4(logits)
        output_softmax = self.softmax_out(output_logits)
        
        return output_logits, output_softmax
    
    def resize(self, ft_input):
        ret = torch.zeros((ft_input.shape[0], self.resnet_num_patches, 1, 224, 224)).cuda()
        
        for i in range(self.resnet_num_patches):
            ret[:,i,:,:,:] = self.transform(ft_input[:,i,:,:,:])
        
        return ret