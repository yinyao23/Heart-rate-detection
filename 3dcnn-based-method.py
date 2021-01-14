import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
import json
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from pre_processing.utils import build_bandpass_filter
from models.cnn3d import CNN

device = torch.device("cpu")
torch.manual_seed(1)  # Set the same random vector
log_file = open('./logs.txt', 'a')
n_plot = 1
scaler = StandardScaler()
FPS = 61
WINDOW_LEN = FPS * 10

def draw_picture(predictions, targets):
  global n_plot
  plt.figure(figsize=(15, 12))
  plt.tick_params(labelsize=23)

  assert len(predictions) == len(targets)
  plot_len = min(1000, len(predictions))
  predictions = predictions[:plot_len]
  targets = targets[:plot_len]
  x = [i for i in range(len(predictions))]
  plt.plot(x, predictions, color='red', linewidth=2, linestyle='-')
  plt.plot(x, targets, color='green', linewidth=2, linestyle='-')
  plt.savefig("results/lstm_results_compare/mean_hr_" + str(n_plot) + '.jpg')
  n_plot += 1
  plt.show()
  plt.close()

def construct_HCI_dataset_for_3dcnn_model():
  '''
  Input: pre-processed dataset file in json format
  Function:
    Bandpass filter for each(pixel, channel) pair in each bounding box sequence
    Apply sliding window strategy to create the dataset
  '''
  groundtruth_file = './dataset/peak_timestamps_groundtruth_hci_tagging.json'
  instant_hr_file = './dataset/instan_hr_groundtruth_hci_tagging.json'
  with open(groundtruth_file, 'r') as file:
    hci_ground_truth = json.load(file)
  with open(instant_hr_file, 'r') as file:
    instant_hr = json.load(file)

  sequences = {}
  bd_files = os.listdir('./dataset/roi_sequence/')
  if '.DS_Store' in bd_files:
    bd_files.remove('.DS_Store')
  for bd_file in bd_files:
    with open('./dataset/roi_sequence/' + bd_file, 'r') as file:
      sequences[bd_file[:-5]] = json.load(file)

  print("All Files:", sequences.keys())
  all_data = []
  for file_idx in sequences.keys():
    sequence = np.array(sequences[file_idx][file_idx])
    # print('Processed vidoe length: {0} minutes'.format(sequence.shape[0] / (FPS * 60)))
    sequence = np.transpose(sequence, (0, 3, 1, 2))
    avg_pool = nn.AdaptiveAvgPool2d((36, 36))
    sequence = avg_pool(torch.from_numpy(sequence).float())

    from scipy.signal import filtfilt
    order = 128
    b = build_bandpass_filter(FPS, order, False)

    _, _, w, h = sequence.size()
    for i in range(w):
      for j in range(h):
        for c in range(3):
          sequence[:, c, i, j] = torch.from_numpy(filtfilt(b, np.array([1]), sequence[:, c, i, j], axis=0).copy())

    # Make the input seq and target seq to have the same length
    timestamps = hci_ground_truth[file_idx]
    instan_hr = instant_hr[file_idx]
    assert len(timestamps) == len(instan_hr)
    total_length = min(len(sequence), timestamps[-1])
    sequence = sequence[: total_length]
    timestamps = [i for i in timestamps if i <= total_length]
    instan_hr = instan_hr[: len(timestamps)]

    # Compute the mean hr seq for every 10 seconds signals from the peak timestamps as groundtruth
    target = []
    for start_point in range(0, total_length - WINDOW_LEN, FPS):
      time_window = [i for i in timestamps if i >= start_point and i < start_point+WINDOW_LEN]
      l_idx = timestamps.index(time_window[0])
      r_idx = timestamps.index(time_window[-1])
      target.append(np.mean(np.array(instan_hr[l_idx: r_idx+1])))

    # apply the sliding window strategy to construct the dataset
    sequence = sequence.tolist()
    seq_windows = [[sequence[i: i + WINDOW_LEN], target[int(i/FPS)]] for i in range(0, total_length - WINDOW_LEN, FPS)]
    # print(seq_windows[0])
    all_data.extend(seq_windows)
  return all_data

def test_accuracy(model, test_dataset):
  '''
  Function: Evaluate the model on all test samples by mean error, RMSE and outlier percent
  '''

  test_batch_size = 1
  test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)
  model.eval()
  # scaler = StandardScaler()
  with torch.no_grad():
    err_list = []
    rmse_list = []
    cor_list = []
    predictions_list = []
    targets_list = []
    for sequence, targets in test_loader:
      # sequence = torch.from_numpy(np.array([scaler.fit_transform(x) for x in sequence])).float()
      predictions = model(sequence)

      # The results are scalers while the code is for vector output
      predictions = predictions.numpy().reshape(-1)
      targets = targets.numpy().reshape(-1)
      errs = abs(predictions - targets)
      err = np.mean(errs)
      rmse = np.sqrt(np.mean(errs**2))
      # print(predictions, targets)

      predictions_list.extend(predictions.tolist())
      targets_list.extend(targets.tolist())
      err_list.append(err)
      rmse_list.append(rmse)
    draw_picture(predictions_list, targets_list)
    print(predictions_list)
    print(targets_list)
    cor = np.corrcoef(np.array(predictions_list), np.array(targets_list))[0][1]
    cor_list.append(cor)
    print("Best trail test set err: {0} rmse: {1} correlation coefficient: {2} \n".format(np.mean(np.array(err_list)), np.mean(np.array(rmse_list)), np.mean(np.array(cor_list))))
    log_file.write("Best trail test set err: {0} rmse: {1} correlation coefficient: {2} \n".format(np.mean(np.array(err_list)), np.mean(np.array(rmse_list)), np.mean(np.array(cor_list))))

def model_train(config, train_dataset, checkpoint_dir=None):
  test_abs = int(len(train_dataset) * 0.8)
  train_subset, val_subset = random_split(train_dataset, [test_abs, len(train_dataset) - test_abs])

  train_loader = DataLoader(dataset=train_subset, batch_size=int(config["batch_size"]), shuffle=True)
  val_loader = DataLoader(dataset=val_subset, batch_size=int(config["batch_size"]), shuffle=True)

  '''Define the CNN model'''
  model = CNN()
  # print(model.state_dict())
  loss_function = nn.MSELoss() #(reduction='mean')
  optimizer = torch.optim.Adam(model.parameters(), lr=3 * pow(10, -3)) # 3 * pow(10, -5)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

  if checkpoint_dir:
    model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)

  '''Model training'''
  model.train()
  for epoch in range(config["n_epoch"]):
    running_loss = []
    for i, data in enumerate(train_loader, 0):
      x_batch, y_batch = data
      # print(x_batch.size(), y_batch.size())

      predictions = model(x_batch)
      # print(predictions.size(), y_batch.size())
      loss = loss_function(predictions, y_batch)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      running_loss.append(loss.item())

      # if i % 100 == 99:  # print every 200 mini-batches
      #   print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, np.mean(np.array(running_loss))))
      #   running_loss = []

    val_loss = []
    mean_error = 0
    model.eval()
    with torch.no_grad():
      predictions_list = []
      targets_list = []
      for i, data in enumerate(val_loader, 0):
        x_batch, y_batch = data
        predictions = model(x_batch)
        # print(predictions.size(), y_batch.size())
        loss = loss_function(predictions, y_batch)
        predictions = predictions.numpy().reshape(-1)
        y_batch = y_batch.numpy().reshape(-1)
        predictions_list.extend(predictions.tolist())
        targets_list.extend(y_batch.tolist())
        mean_error += np.mean(abs(predictions-y_batch))
        val_loss.append(loss.item())
    scheduler.step(sum(val_loss))

    print('epoch: {0} Training loss: {1} Validation loss: {2}'.format(epoch, np.mean(np.array(running_loss)), np.mean(np.array(val_loss))))
    log_file.write('epoch: {0} Training loss: {1} Validation loss: {2}\n'.format(epoch, np.mean(np.array(running_loss)), np.mean(np.array(val_loss))))
  torch.save((model.state_dict(), optimizer.state_dict()), './models/best_model.pt')
  return model



if __name__ == "__main__":
  torch.autograd.set_detect_anomaly(True)
  train_test_ratio = 0.6
  config = {
    "n_epoch": 10,
    "batch_size": 1
  }

  all_data = construct_HCI_dataset_for_3dcnn_model()
  x_appearance = np.array([x[0] for x in all_data])
  x_motion = []
  for window in x_appearance:
    t = window[1] - window[0]
    diff = [window[i] - window[i-1] for i in range(1, len(window))]
    x_motion.append(diff)
  x_appearance = np.transpose(x_appearance[:, :-1, :, :, :], (0, 2, 1, 3, 4))
  x_motion = np.transpose(np.array(x_motion), (0, 2, 1, 3, 4))
  # print(x_appearance.shape, x_motion.shape)
  x_tensor = torch.from_numpy(np.stack((x_motion, x_appearance), axis=1)).float()
  y_tensor = torch.from_numpy(np.array([[x[1]] for x in all_data])).float()
  # print(x_tensor.size(), y_tensor.size())
  dataset = TensorDataset(x_tensor, y_tensor)

  train_num = int(train_test_ratio * (len(dataset)))
  train_dataset, test_dataset = random_split(dataset, [train_num, len(dataset)-train_num])
  best_trained_model = model_train(config=config, train_dataset=train_dataset)

  print("******** Final Testing result ********")
  test_accuracy(best_trained_model, test_dataset)
  log_file.close()