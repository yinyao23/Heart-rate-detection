import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from ray import tune
from ray.tune import CLIReporter
from functools import partial
from torch.utils.data import TensorDataset

import json
import numpy as np
import os

from models.cnn1d import CNN
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import matplotlib.style as style
style.use("ggplot")

device = torch.device("cpu")
torch.manual_seed(1)  # Set the same random vector
log_file = open('./logs.txt', 'a')
n_plot = 1
scaler = StandardScaler()

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

def test_accuracy(model, test_dataset):
  '''
  Function: Evaluate the model on all test samples by mean error, RMSE and outlier percent
  '''

  test_batch_size = 1
  test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)
  model.eval()
  scaler = StandardScaler()
  with torch.no_grad():
    err_list = []
    rmse_list = []
    cor_list = []
    predictions_list = []
    targets_list = []
    for sequence, targets in test_loader:
      # print(sequence.size(), targets.size())
      sequence = torch.from_numpy(np.array([scaler.fit_transform(x) for x in sequence])).float()
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
  # print(config)
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

    val_loss = 0.0
    val_steps = 0
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
        val_loss += loss.item()
        val_steps += 1
    scheduler.step(val_loss)

    with tune.checkpoint_dir(epoch) as checkpoint_dir:
      path = os.path.join(checkpoint_dir, "checkpoint")
      torch.save((model.state_dict(), optimizer.state_dict()), path)
    tune.report(train_loss=np.mean(np.array(running_loss)), loss=(val_loss / val_steps), mean_error=mean_error / val_steps)

if __name__ == "__main__":
  torch.autograd.set_detect_anomaly(True)
  train_test_ratio = 0.6
  config = {
    "n_epoch": tune.grid_search([200]),
    "batch_size": tune.grid_search([32])
  }

  with open('./dataset/processed_windows_hci_tagging.json', 'r') as file:
    all_data = json.load(file)['windows']
    all_data = all_data[: int(len(all_data))]
    x_tensor = torch.from_numpy(np.array([[[i[1]] for i in x[0]] for x in all_data])).float()
    y_tensor = torch.from_numpy(np.array([[x[1]] for x in all_data])).float()
    print(x_tensor.size(), y_tensor.size())
    dataset = TensorDataset(x_tensor, y_tensor)

  # Validate the correaltion between input and output data
  # all_data = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
  # predictions = []
  # targets = []
  # for i, data in enumerate(all_data, 0):
  #   # print(data)
  #   transformed = data[0].numpy().reshape(-1, 3)
  #   powerSpec = np.abs(np.fft.fft(transformed, axis=0)) ** 2
  #   maxPwrSrc = np.max(powerSpec, axis=1)
  #   freqs = np.fft.fftfreq(len(transformed), 1.0 / 61)
  #
  #   '''Filter the HR signals using the frequency band'''
  #   valid_idx = np.where((freqs >= 60 / 60) & (freqs <= 240 / 60))
  #   valid_pwr = maxPwrSrc[valid_idx]
  #   valid_freqs = freqs[valid_idx]
  #
  #   max_pwr_idx = np.argmax(valid_pwr)
  #   predictions.append(valid_freqs[max_pwr_idx] * 60.0)
  #   targets.append(data[1].numpy().tolist()[0][0])
  # print(predictions)
  # print(targets)
  # cor = np.corrcoef(np.array(predictions), np.array(targets))[0][1]
  # print("Correlation between input data and targets:", cor)
  # from scipy import signal
  # cross_corr = signal.correlate(predictions, targets, mode='same')
  # print(len(cross_corr), cross_corr)
  # plt.plot(cross_corr, label='cross-correlation')
  # plt.show()

  train_num = int(train_test_ratio * (len(dataset)))
  train_dataset, test_dataset = random_split(dataset, [train_num, len(dataset)-train_num])

  reporter = CLIReporter(
    metric_columns=["train_loss", "loss", "mean_error", "training_iteration"])
  results = tune.run(
    partial(model_train, train_dataset=train_dataset),
    # resources_per_trial={"cpu": 2},
    config=config,
    progress_reporter=reporter)

  # Plot loss during training
  dfs = results.fetch_trial_dataframes()
  [d.train_loss.plot() for d in dfs.values()]
  [d.loss.plot() for d in dfs.values()]
  plt.xlabel("epoch")

  best_trial = results.get_best_trial("loss", "min", "last") # "last"
  log_file.write("Best trial config: {}".format(best_trial.config))
  log_file.write("Best trial final validation loss: {}".format(
    best_trial.last_result["loss"]))
  log_file.write("Best trial final validation mean error: {}".format(
    best_trial.last_result["mean_error"]))

  best_trained_model = CNN()
  best_checkpoint_dir = best_trial.checkpoint.value
  model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
  best_trained_model.load_state_dict(model_state)

  print("******** Final Testing result ********")
  test_accuracy(best_trained_model, test_dataset)
  log_file.close()