import torch
import torch.nn as nn
import copy

class CNN(nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
      super(CNN, self).__init__()
      self.m_cnn1 = nn.Sequential(
        nn.Conv3d(input_dim, 16, kernel_size=(3, 3, 3)),
        nn.BatchNorm3d(16),
        nn.ReLU(),

        nn.Conv3d(16, 16, kernel_size=(3, 3, 3)),
        nn.BatchNorm3d(16),
        nn.ReLU(),

        nn.Dropout3d(p=0.25),
      )

      self.m_cnn2 = nn.Sequential(
        nn.AvgPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2)),

        nn.Conv3d(16, 32, kernel_size=(5, 3, 3)),
        nn.BatchNorm3d(32),
        nn.ReLU(),

        nn.Conv3d(32, 32, kernel_size=(5, 3, 3)),
        nn.BatchNorm3d(32),
        nn.ReLU(),

        nn.Dropout3d(p=0.25)
      )

      self.a_cnn1 = copy.deepcopy(self.m_cnn1)
      self.a_cnn2 = copy.deepcopy(self.m_cnn2)

      self.attn1 = nn.Sequential(
        nn.Conv3d(16, 16, kernel_size=(1, 1, 1)),
        nn.Sigmoid(),
        nn.BatchNorm3d(16),
      )

      self.attn2 = nn.Sequential(
        nn.Conv3d(32, 32, kernel_size=(1, 1, 1)),
        nn.Sigmoid(),
        nn.BatchNorm3d(32),
      )

      self.global_avg_pool = nn.AvgPool3d(kernel_size=(1, 11, 11))
      self.fc1 = nn.Linear(19040, 1024)
      nn.init.xavier_normal_(self.fc1.weight)
      self.tanh1 = nn.Tanh()
      self.fc2 = nn.Linear(1024, 64)
      nn.init.xavier_normal_(self.fc2.weight)
      self.tanh2 = nn.Tanh()
      self.fc3 = nn.Linear(64, output_dim)
      nn.init.xavier_normal_(self.fc3.weight)


    def forward(self, x):
      A = x[:, 1, :, :, :, :].squeeze(1)
      M = x[:, 0, :, :, :, :].squeeze(1)
      # print(M.size(), A.size())

      A = self.a_cnn1(A)
      # Calculating attention roi1 with soft-attention1
      roi1 = self.attn1(A)
      B, _, _, H, W = A.shape
      norm = 2 * torch.norm(roi1, p=1, dim=(1, 2, 3, 4))
      norm = norm.reshape(B, 1, 1, 1, 1)
      roi1 = torch.div(roi1 * H * W, norm)

      A = self.a_cnn2(A)
      # Calculating attention roi2 with soft-attention2
      roi2 = self.attn2(A)
      B, _, _, H, W = A.shape
      norm = 2 * torch.norm(roi2, p=1, dim=(1, 2, 3, 4))
      norm = norm.reshape(B, 1, 1, 1, 1)
      roi2 = torch.div(roi2 * H * W, norm)

      # print(M.size(), roi1.size(), roi2.size())
      M = self.m_cnn1(M)
      M = torch.tanh(torch.mul(M, roi1))
      M = self.m_cnn2(M)
      M = torch.tanh(torch.mul(M, roi2))
      M = self.global_avg_pool(M)

      # flat the M tensor
      # print(M.size())
      M = M.squeeze(4).squeeze(3)
      M = M.reshape((M.size()[0], -1))
      # print(M.size())

      M = self.fc1(M)
      M = self.tanh1(M)
      M = self.fc2(M)
      M = self.tanh2(M)
      M = self.fc3(M)
      # print(M.size())
      return M

if __name__ == "__main__":
  x = torch.randn(32, 2, 3, 609, 36, 36)
  model = CNN()
  output = model(x)
  print(output.size())



