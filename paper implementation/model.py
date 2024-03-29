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
import torch.nn.functional as F
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

class transfer_resnet18(nn.Module):
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
        print(ft_embedding.shape)
        
        output_logits = self.fc_linear(ft_embedding)
        output_softmax = self.softmax_out(output_logits)
        
        return output_logits, output_softmax
    
    def resize(self, ft_input):
        ret = torch.zeros((ft_input.shape[0], self.resnet_num_patches, 1, 224, 224)).cuda()
        
        for i in range(self.resnet_num_patches):
            ret[:,i,:,:,:] = self.transform(ft_input[:,i,:,:,:])
        
        return ret    

class gru_lstm_transformer_transfer_alexnet(nn.Module):
    def __init__(self, num_emotions) -> None:
        super().__init__()
        
        self.transform = transforms.Resize([224,224])
        self.resnet_patch_size = 40
        self.resnet_num_patches = 7
        
        self.model_ft1 = models.alexnet(pretrained=True)
        self.model_ft1 = torch.nn.Sequential(*(list(self.model_ft1.children())[:-1]))
        self.model_ft2 = models.alexnet(pretrained=True)
        self.model_ft2 = torch.nn.Sequential(*(list(self.model_ft2.children())[:-1]))
        self.model_ft3 = models.alexnet(pretrained=True)
        self.model_ft3 = torch.nn.Sequential(*(list(self.model_ft3.children())[:-1]))
        self.model_ft4 = models.alexnet(pretrained=True)
        self.model_ft4 = torch.nn.Sequential(*(list(self.model_ft4.children())[:-1]))
        self.model_ft5 = models.alexnet(pretrained=True)
        self.model_ft5 = torch.nn.Sequential(*(list(self.model_ft5.children())[:-1]))
        self.model_ft6 = models.alexnet(pretrained=True)
        self.model_ft6 = torch.nn.Sequential(*(list(self.model_ft6.children())[:-1]))
        self.model_ft7 = models.alexnet(pretrained=True)
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
        
        self.fc_linear1 = nn.Linear(65576, 1024)
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
        ft_output1 = torch.flatten(ft_output1, start_dim=1)     # (32, 9216)
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


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
    
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
    
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x

class Cnn14(nn.Module):
    def __init__(self):
        
        super(Cnn14, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
    
    def forward(self, x):
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.5)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.5)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.5)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.5)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.5)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.5)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5)
        
        return embedding  

class gru_lstm_transformer_transfer_cnn14(nn.Module):
    def __init__(self, num_emotions) -> None:
        super().__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=[1, 4], stride=[1, 4])

        self.relu = nn.ReLU()

        self.gru = nn.GRU(input_size=40, hidden_size=512, num_layers=4, batch_first=True, bidirectional=True, dropout=0.5)
        self.gru_ln = nn.LayerNorm(normalized_shape=1024, eps=1e-08)
        
        self.lstm = nn.LSTM(input_size=40, hidden_size=512, num_layers=4, batch_first=True, bidirectional=True, dropout=0.5)
        self.lstm_ln = nn.LayerNorm(normalized_shape=1024, eps=1e-08)

        transformer_layer = nn.TransformerEncoderLayer(
            # input feature (frequency) dim after maxpooling 40*282 -> 40*70 (MFC*time)
            d_model=40,
            nhead=4,  # 4 self-attention layers in each multi-head self-attention layer in each encoder block
            # 2 linear layers in each encoder block's feedforward network: dim 40-->512--->40
            dim_feedforward=512,
            dropout=0.5,
            activation='relu'  # ReLU: avoid saturation/tame gradient/reduce compute time
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, num_layers=4)
        self.transformer_ln = nn.LayerNorm(normalized_shape=40, eps=1e-08)
    
        self.cnn14 = Cnn14()
        checkpoint = torch.load('pth/Cnn14_16k.pth', map_location='cuda')     # 모델을 동적으로 GPU에 할당
        self.cnn14.load_state_dict(checkpoint['model'], strict=False)         # 더 많은 키를 갖고 있는 경우 strict=False
        
        
        self.fc_linear1 = nn.Linear(4136, 1024)
        self.fc_linear2 = nn.Linear(1024, 512)
        self.fc_linear3 = nn.Linear(512, 256) 
        self.fc_linear4 = nn.Linear(256, num_emotions)
        self.softmax_out = nn.Softmax(dim=1)
        
    
    def forward(self, x):
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
        
        cnn14_embedding = self.cnn14(x)
        
        complete_embedding = torch.cat([cnn14_embedding, gru_embedding, lstm_embedding, transformer_embedding], dim=1)
        
        logits = self.fc_linear1(complete_embedding)
        logits = self.fc_linear2(logits)
        logits = self.fc_linear3(logits)
        output_logits = self.fc_linear4(logits)
        output_softmax = self.softmax_out(output_logits)
        
        return output_logits, output_softmax

class Cnn14(nn.Module):
    def __init__(self):
        
        super(Cnn14, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
    
    def forward(self, x):
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.5)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.5)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.5)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.5)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.5)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.5)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5)
        
        return embedding  

class transfer_cnn14(nn.Module):
    def __init__(self, num_emotions) -> None:
        super().__init__()
    
        self.cnn14 = Cnn14()
        checkpoint = torch.load('pth/Cnn14_16k.pth', map_location='cuda')     # 모델을 동적으로 GPU에 할당
        self.cnn14.load_state_dict(checkpoint['model'], strict=False)         # 더 많은 키를 갖고 있는 경우 strict=False
        
        self.fc_linear = nn.Linear(512, num_emotions)
        self.softmax_out = nn.Softmax(dim=1)
        
    
    def forward(self, x):
        cnn14_embedding = self.cnn14(x)
        print(cnn14_embedding)
        
        output_logits = self.fc_linear(cnn14_embedding)
        output_softmax = self.softmax_out(output_logits)
        
        return output_logits, output_softmax    

class MobileNetV1(nn.Module):
    def __init__(self):
        
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False), 
                nn.AvgPool2d(stride), 
                nn.BatchNorm2d(oup), 
                nn.ReLU(inplace=True)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            return _layers

        def conv_dw(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False), 
                nn.AvgPool2d(stride), 
                nn.BatchNorm2d(inp), 
                nn.ReLU(inplace=True), 
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False), 
                nn.BatchNorm2d(oup), 
                nn.ReLU(inplace=True)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            init_layer(_layers[4])
            init_bn(_layers[5])
            return _layers

        self.features = nn.Sequential(
            conv_bn(  1,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1))

        self.fc1 = nn.Linear(1024, 1024, bias=True)

        init_layer(self.fc1)
 
    def forward(self, x):
        x = self.features(x)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)

        return embedding

class gru_lstm_transformer_transfer_MobileNetV1(nn.Module):
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
    
        self.mobilenetv1 = MobileNetV1()
        checkpoint = torch.load('pth/MobileNetV1.pth', map_location='cuda')     # 모델을 동적으로 GPU에 할당
        self.mobilenetv1.load_state_dict(checkpoint['model'], strict=False)         # 더 많은 키를 갖고 있는 경우 strict=False
        
        
        self.fc_linear1 = nn.Linear(3112, 1024)
        self.fc_linear2 = nn.Linear(1024, 512)
        self.fc_linear3 = nn.Linear(512, 256) 
        self.fc_linear4 = nn.Linear(256, num_emotions)
        self.softmax_out = nn.Softmax(dim=1)
        
    
    def forward(self, x):
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
        
        mobilenetV1_embedding = self.mobilenetv1(x)
        
        complete_embedding = torch.cat([mobilenetV1_embedding, gru_embedding, lstm_embedding, transformer_embedding], dim=1)
        
        logits = self.fc_linear1(complete_embedding)
        logits = self.fc_linear2(logits)
        logits = self.fc_linear3(logits)
        output_logits = self.fc_linear4(logits)
        output_softmax = self.softmax_out(output_logits)
        
        return output_logits, output_softmax
    

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            _layers = [
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False), 
                nn.AvgPool2d(stride), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU6(inplace=True), 
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), 
                nn.BatchNorm2d(oup)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            init_layer(_layers[4])
            init_bn(_layers[5])
            self.conv = _layers
        else:
            _layers = [
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU6(inplace=True), 
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False), 
                nn.AvgPool2d(stride), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU6(inplace=True), 
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), 
                nn.BatchNorm2d(oup)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[1])
            init_layer(_layers[3])
            init_bn(_layers[5])
            init_layer(_layers[7])
            init_bn(_layers[8])
            self.conv = _layers

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
class MobileNetV2(nn.Module):
    def __init__(self):
        
        super(MobileNetV2, self).__init__()
 
        width_mult=1.
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 2],
            [6, 160, 3, 1],
            [6, 320, 1, 1],
        ]

        def conv_bn(inp, oup, stride):
            _layers = [
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False), 
                nn.AvgPool2d(stride), 
                nn.BatchNorm2d(oup), 
                nn.ReLU6(inplace=True)
                ]
            _layers = nn.Sequential(*_layers)
            init_layer(_layers[0])
            init_bn(_layers[2])
            return _layers


        def conv_1x1_bn(inp, oup):
            _layers = nn.Sequential(
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
            init_layer(_layers[0])
            init_bn(_layers[1])
            return _layers

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self.fc1 = nn.Linear(1280, 1024, bias=True)
        
        init_layer(self.fc1)
 
    def forward(self, x):

        x = self.features(x)
        
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        
        return embedding

class gru_lstm_transformer_transfer_MobileNetV2(nn.Module):
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
    
        self.mobilenetv2 = MobileNetV2()
        checkpoint = torch.load('pth/MobileNetV2.pth', map_location='cuda')     # 모델을 동적으로 GPU에 할당
        self.mobilenetv2.load_state_dict(checkpoint['model'], strict=False)         # 더 많은 키를 갖고 있는 경우 strict=False
        
        
        self.fc_linear1 = nn.Linear(3112, 1024)
        self.fc_linear2 = nn.Linear(1024, 512)
        self.fc_linear3 = nn.Linear(512, 256) 
        self.fc_linear4 = nn.Linear(256, num_emotions)
        self.softmax_out = nn.Softmax(dim=1)
        
    
    def forward(self, x):
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
        
        mobilenetV2_embedding = self.mobilenetv2(x)
        
        complete_embedding = torch.cat([mobilenetV2_embedding, gru_embedding, lstm_embedding, transformer_embedding], dim=1)
        
        logits = self.fc_linear1(complete_embedding)
        logits = self.fc_linear2(logits)
        logits = self.fc_linear3(logits)
        output_logits = self.fc_linear4(logits)
        output_softmax = self.softmax_out(output_logits)
        
        return output_logits, output_softmax
    

def _resnet_conv3x3(in_planes, out_planes):
    #3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)

def _resnet_conv1x1(in_planes, out_planes):
    #1x1 convolution
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)

class _ResnetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('_ResnetBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in _ResnetBasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.stride = stride

        self.conv1 = _resnet_conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _resnet_conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            out = F.avg_pool2d(x, kernel_size=(2, 2))
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class _ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(_ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[0])
                init_bn(downsample[1])
            elif stride == 2:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2), 
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[1])
                init_bn(downsample[2])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    
class ResNet38(nn.Module):
    def __init__(self):
        
        super(ResNet38, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        # self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)

        self.resnet = _ResNet(block=_ResnetBasicBlock, layers=[3, 4, 6, 3], zero_init_residual=True)

        self.conv_block_after1 = ConvBlock(in_channels=512, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048)

        init_layer(self.fc1)

    def forward(self, x):
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)

        return embedding

class gru_lstm_transformer_transfer_ResNet38(nn.Module):
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
    
        self.resnet38 = ResNet38()
        checkpoint = torch.load('pth/ResNet38.pth', map_location='cuda')     # 모델을 동적으로 GPU에 할당
        self.resnet38.load_state_dict(checkpoint['model'], strict=False)         # 더 많은 키를 갖고 있는 경우 strict=False
        
        
        self.fc_linear1 = nn.Linear(4136, 1024)
        self.fc_linear2 = nn.Linear(1024, 512)
        self.fc_linear3 = nn.Linear(512, 256) 
        self.fc_linear4 = nn.Linear(256, num_emotions)
        self.softmax_out = nn.Softmax(dim=1)
        
    
    def forward(self, x):
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
        
        resnet38_embedding = self.resnet38(x)
        
        complete_embedding = torch.cat([resnet38_embedding, gru_embedding, lstm_embedding, transformer_embedding], dim=1)
        
        logits = self.fc_linear1(complete_embedding)
        logits = self.fc_linear2(logits)
        logits = self.fc_linear3(logits)
        output_logits = self.fc_linear4(logits)
        output_softmax = self.softmax_out(output_logits)
        
        return output_logits, output_softmax

class transfer_ResNet38(nn.Module):
    def __init__(self, num_emotions) -> None:
        super().__init__()
        
        self.resnet38 = ResNet38()
        checkpoint = torch.load('pth/ResNet38.pth', map_location='cuda')     # 모델을 동적으로 GPU에 할당
        self.resnet38.load_state_dict(checkpoint['model'], strict=False)         # 더 많은 키를 갖고 있는 경우 strict=False
        
        
        self.fc_linear = nn.Linear(256, num_emotions)
        self.softmax_out = nn.Softmax(dim=1)
        
    
    def forward(self, x):
        resnet38_embedding = self.resnet38(x)
        print(resnet38_embedding.shape)
        
        output_logits = self.fc_linear(resnet38_embedding)
        output_softmax = self.softmax_out(output_logits)
        
        return output_logits, output_softmax

class _ResnetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.stride = stride
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = _resnet_conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = _resnet_conv3x3(width, width)
        self.bn2 = norm_layer(width)
        self.conv3 = _resnet_conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        init_layer(self.conv3)
        init_bn(self.bn3)
        nn.init.constant_(self.bn3.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            x = F.avg_pool2d(x, kernel_size=(2, 2))

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class ResNet54(nn.Module):
    def __init__(self):
        
        super(ResNet54, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        # self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)

        self.resnet = _ResNet(block=_ResnetBottleneck, layers=[3, 4, 6, 3], zero_init_residual=True)

        self.conv_block_after1 = ConvBlock(in_channels=2048, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048)

        init_layer(self.fc1)

    def forward(self, x):
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        
        return embedding
    
class gru_lstm_transformer_transfer_ResNet54(nn.Module):
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
    
        self.resnet54 = ResNet54()
        checkpoint = torch.load('pth/ResNet54.pth', map_location='cuda')     # 모델을 동적으로 GPU에 할당
        self.resnet54.load_state_dict(checkpoint['model'], strict=False)         # 더 많은 키를 갖고 있는 경우 strict=False
        
        
        self.fc_linear1 = nn.Linear(4136, 1024)
        self.fc_linear2 = nn.Linear(1024, 512)
        self.fc_linear3 = nn.Linear(512, 256) 
        self.fc_linear4 = nn.Linear(256, num_emotions)
        self.softmax_out = nn.Softmax(dim=1)
        
    
    def forward(self, x):
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
        
        resnet54_embedding = self.resnet54(x)
        
        complete_embedding = torch.cat([resnet54_embedding, gru_embedding, lstm_embedding, transformer_embedding], dim=1)
        
        logits = self.fc_linear1(complete_embedding)
        logits = self.fc_linear2(logits)
        logits = self.fc_linear3(logits)
        output_logits = self.fc_linear4(logits)
        output_softmax = self.softmax_out(output_logits)
        
        return output_logits, output_softmax
    
from torchvision.models import AlexNet
class gru_lstm_transformer_transfer_AlexNet(nn.Module):
    def __init__(self, num_emotions) -> None:
        super().__init__()
        
        self.transform = transforms.Resize([224,224])
        self.resnet_patch_size = 40
        self.resnet_num_patches = 7
        
        checkpoint = 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth'
        
        self.model_ft1 = AlexNet()
        self.model_ft1.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False), strict=False)
        self.model_ft1 = torch.nn.Sequential(*(list(self.model_ft1.children())[:-1]))
        self.model_ft1 = nn.Sequential(
            *(list(self.model_ft1.children())[:-1]),
            nn.Dropout(0.4)           
        )
        
        self.model_ft2 = AlexNet()
        self.model_ft2.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False), strict=False)
        self.model_ft2 = torch.nn.Sequential(*(list(self.model_ft2.children())[:-1]))
        self.model_ft2 = nn.Sequential(
            *(list(self.model_ft2.children())[:-1]),
            nn.Dropout(0.4)           
        )
        
        self.model_ft3 = AlexNet()
        self.model_ft3.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False), strict=False)
        self.model_ft3 = torch.nn.Sequential(*(list(self.model_ft3.children())[:-1]))
        self.model_ft3 = nn.Sequential(
            *(list(self.model_ft3.children())[:-1]),
            nn.Dropout(0.4)           
        )
        
        self.model_ft4 = AlexNet()
        self.model_ft4.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False), strict=False)
        self.model_ft4 = torch.nn.Sequential(*(list(self.model_ft4.children())[:-1]))
        self.model_ft4 = nn.Sequential(
            *(list(self.model_ft4.children())[:-1]),
            nn.Dropout(0.4)           
        )

        self.model_ft5 = AlexNet()
        self.model_ft5.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False), strict=False)
        self.model_ft5 = torch.nn.Sequential(*(list(self.model_ft5.children())[:-1]))
        self.model_ft5 = nn.Sequential(
            *(list(self.model_ft5.children())[:-1]),
            nn.Dropout(0.4)           
        )
        
        self.model_ft6 = AlexNet()
        self.model_ft6.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False), strict=False)
        self.model_ft6 = torch.nn.Sequential(*(list(self.model_ft6.children())[:-1]))
        self.model_ft6 = nn.Sequential(
            *(list(self.model_ft6.children())[:-1]),
            nn.Dropout(0.4)           
        )
        
        self.model_ft7 = AlexNet()
        self.model_ft7.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False), strict=False)
        self.model_ft7 = torch.nn.Sequential(*(list(self.model_ft7.children())[:-1]))
        self.model_ft7 = nn.Sequential(
            *(list(self.model_ft7.children())[:-1]),
            nn.Dropout(0.4)           
        )
        
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
        
        self.fc_linear1 = nn.Linear(65576, 1024)
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
        # print(complete_embedding.shape)
        
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
    
class transfer_vgg11_bn(nn.Module):
    def __init__(self, num_emotions) -> None:
        super().__init__()
        
        self.transform = transforms.Resize([224,224])
        self.resnet_patch_size = 40
        self.resnet_num_patches = 7
        
        self.model_ft1 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=True)
        self.model_ft1 = torch.nn.Sequential(*(list(self.model_ft1.children())[:-1]))
        self.model_ft2 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=True)
        self.model_ft2 = torch.nn.Sequential(*(list(self.model_ft2.children())[:-1]))
        self.model_ft3 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=True)
        self.model_ft3 = torch.nn.Sequential(*(list(self.model_ft3.children())[:-1]))
        self.model_ft4 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=True)
        self.model_ft4 = torch.nn.Sequential(*(list(self.model_ft4.children())[:-1]))
        self.model_ft5 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=True)
        self.model_ft5 = torch.nn.Sequential(*(list(self.model_ft5.children())[:-1]))
        self.model_ft6 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=True)
        self.model_ft6 = torch.nn.Sequential(*(list(self.model_ft6.children())[:-1]))
        self.model_ft7 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn', pretrained=True)
        self.model_ft7 = torch.nn.Sequential(*(list(self.model_ft7.children())[:-1]))
        
        self.fc_linear = nn.Linear(175616, num_emotions)
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
        
        output_logits = self.fc_linear(ft_embedding)
        output_softmax = self.softmax_out(output_logits)
        
        return output_logits, output_softmax
    
    def resize(self, ft_input):
        ret = torch.zeros((ft_input.shape[0], self.resnet_num_patches, 1, 224, 224)).cuda()
        
        for i in range(self.resnet_num_patches):
            ret[:,i,:,:,:] = self.transform(ft_input[:,i,:,:,:])
        
        return ret

class transfer_densenet121(nn.Module):
    def __init__(self, num_emotions) -> None:
        super().__init__()
        
        self.transform = transforms.Resize([224,224])
        self.resnet_patch_size = 40
        self.resnet_num_patches = 7
        
        self.model_ft1 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.model_ft1 = torch.nn.Sequential(*(list(self.model_ft1.children())[:-1]))
        self.model_ft2 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.model_ft2 = torch.nn.Sequential(*(list(self.model_ft2.children())[:-1]))
        self.model_ft3 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.model_ft3 = torch.nn.Sequential(*(list(self.model_ft3.children())[:-1]))
        self.model_ft4 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.model_ft4 = torch.nn.Sequential(*(list(self.model_ft4.children())[:-1]))
        self.model_ft5 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.model_ft5 = torch.nn.Sequential(*(list(self.model_ft5.children())[:-1]))
        self.model_ft6 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.model_ft6 = torch.nn.Sequential(*(list(self.model_ft6.children())[:-1]))
        self.model_ft7 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        self.model_ft7 = torch.nn.Sequential(*(list(self.model_ft7.children())[:-1]))
        
        self.fc_linear = nn.Linear(175616, num_emotions)
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
        print(ft_embedding.shape)
        
        output_logits = self.fc_linear(ft_embedding)
        output_softmax = self.softmax_out(output_logits)
        
        return output_logits, output_softmax
    
    def resize(self, ft_input):
        ret = torch.zeros((ft_input.shape[0], self.resnet_num_patches, 1, 224, 224)).cuda()
        
        for i in range(self.resnet_num_patches):
            ret[:,i,:,:,:] = self.transform(ft_input[:,i,:,:,:])
        
        return ret
    
