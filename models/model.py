import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.fc = nn.Linear(configs["model"]["num_hidden_state_features"], 1)
    
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
        output = F.relu(output)
        output = output.permute((0, 2, 1, -1)) #(batch, window_size, num_out_channel, remain)  
        output = torch.flatten(output, start_dim=2, end_dim=3) #4D -> 3D to be used with lstm
        output = self.lstm(output)[0] #(h_n, cn)->hn
        output = torch.select(output, dim=1, index=-1) #get the last hidden state (with num_hidden_state_features)
        output = self.fc(output)
        return output