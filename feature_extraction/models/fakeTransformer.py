import torch
import torch.nn as nn


class FakeTransformer(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FakeTransformer, self).__init__()

        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.dropout1 = nn.Dropout(0.5)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # out = self.fc1(x)
        
        # out = self.relu(out)
        # out = self.fc2(out)
        # out = self.dropout1(out)
        return self.fc(x)
