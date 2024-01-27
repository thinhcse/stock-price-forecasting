import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, feature_size):
        super(SelfAttention, self).__init__()
        self.feature_size = feature_size

        self.key = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)
    
    def forward(self, x, mask = None):
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))

        if mask:
            scores = scores.masked_fill(mask==0, -1e9)
        
        attention_weights = F.softmax(scores, dim = -1)
        output = torch.matmul(attention_weights, values)

        return output, attention_weights
    
class model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs
        self.conv = nn.Conv2d(
                              in_channels=1,
                              out_channels=configs["model"]["num_filters"], 
                              kernel_size= (configs["model"]["kernel_size"], configs["model"]["num_features"]),
                              stride=1,
                              padding=(0, 0)
                              )
        self.lstm = nn.LSTM(
                            input_size=configs["model"]["num_filters"],
                            hidden_size=configs["model"]["num_hidden_state_features"],
                            num_layers=configs["model"]["num_lstm_layers"],
                            batch_first=True
                            )
        self.attention = SelfAttention(feature_size=configs["model"]["num_hidden_state_features"])
        self.fc1 = nn.Linear(configs["model"]["num_hidden_state_features"], 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, input):
        output = input.view((
                                input.shape[0], #flexible batch_size
                                -1, #number of channels (should result in 1)
                                self.configs["model"]["window_size"],
                                self.configs["model"]["num_features"]
                            ))
        output = F.pad(
                      input=output,
                      pad=(
                          0, 0,
                          0, self.configs["model"]["kernel_size"] - 1 #to capture last row when striding
                      ))
        output = self.conv(output)
        output = F.avg_pool2d(output, kernel_size=(2, 1))
        output = output.permute((0, 2, 1, -1)) #(batch, window_size, num_out_channel, remain)  
        output = torch.flatten(output, start_dim=2, end_dim=3) #4D -> 3D to be used with lstm
        output = self.lstm(output)[0] #(h_n, cn)->hn -> feature_size = feature_size(hn)
        output = self.attention(output)[0] #values applied attention, attention weight -> take the first one
        output = torch.select(output, dim=1, index=-1) #get the last hidden state (with num_hidden_state_features)
        output = self.fc1(output)
        output = self.fc2(output)
        return output