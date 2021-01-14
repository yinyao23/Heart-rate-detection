import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_dim=1, output_dim=1):
      super(CNN, self).__init__()
      self.cnn = nn.Sequential(
        nn.Conv1d(input_dim, 16, kernel_size=3),
        nn.ReLU(),

        nn.Conv1d(16, 16, kernel_size=3),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),

        nn.Conv1d(16, 32, kernel_size=3, dilation=2),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        # nn.MaxPool1d(kernel_size=2, stride=2),
        nn.MaxPool1d(kernel_size=299),

        # nn.Conv1d(32, 64, kernel_size=3, dilation=2),
        # nn.BatchNorm1d(64),
        # nn.ReLU(),
        # nn.MaxPool1d(kernel_size=2, stride=2),
        #
        # nn.Conv1d(64, 128, kernel_size=3, dilation=2),
        # nn.BatchNorm1d(128),
        # nn.ReLU(),
        # nn.MaxPool1d(kernel_size=67),
      )

      self.fc1 = nn.Linear(32, 16)
      nn.init.xavier_normal_(self.fc1.weight)
      self.tanh = nn.Tanh()
      self.fc2 = nn.Linear(16, output_dim)
      nn.init.xavier_normal_(self.fc2.weight)


    def forward(self, x):
      # x.size = (batch_size, seq_length, input_dim=3)
      # print(x.size())
      y = self.cnn(x.view(-1, x.size()[2], x.size()[1]))  # output shape([x_len, batch_size, hidden_dim])
      # print(y.size())
      y = self.fc1(y.squeeze(-1))
      y = self.tanh(y)
      y = self.fc2(y)
      # print(y.size())
      return y

# input = torch.randn(16, 610, 3)
# model = 1dCNN()
# output = model(input)
# print(output.size())



