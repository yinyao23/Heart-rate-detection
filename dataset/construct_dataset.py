import numpy as np
import json
import cv2
import os
import dlib
import matplotlib.pyplot as plt
from imutils import face_utils

from .utils import detrend, average
from .utils import build_bandpass_filter
from .utils import capture_video_frames

FPS = 61
WINDOW_LEN = FPS * 10

def construct_HCI_dataset_for_ica_method():
  '''
  Input file: mean_hr_groundtruth_hci_tagging.json (Generated after running preprocess-for-hci-tagging-db.py)
  Output file: mean_rgb_sequence.json (Spatial averaged values on RGB channels over ROIs )
  '''

  with open('./mean_hr_groundtruth_hci_tagging.json') as json_file:
    file_indices = json.load(json_file).keys()
  with open('./mean_rgb_sequence.json') as json_file:
    mean_rgb_sequence = json.load(json_file)

  dir1 = './dataset'  # HCI dataset path
  session_files = os.listdir(dir1)
  if '.DS_Store' in session_files:
    session_files.remove('.DS_Store')
  for session in session_files:
    dir2 = dir1 + '/' + session
    files = os.listdir(dir2)
    if '.DS_Store' in files:
      files.remove('.DS_Store')

    for file in files:
      if file not in file_indices:  # skip files without groundtruth
        print('File {0} does not have the groundtruth data'.format(file))
        continue
      if file in mean_rgb_sequence.keys():
        print('File {0} has already been processed'.format(file))
        continue
      dir3 = dir2 + '/' + file
      print("For file:", file)
      sample = os.listdir(dir3)
      video = [ele for ele in sample if ele[-3:] == 'avi'][0]
      fps, sequence = capture_video_frames(dir3 + '/' + video)
      # print("sequence:", sequence)
      mean_rgb_sequence[file] = sequence
      with open('./mean_rgb_sequence.json', 'w') as outfile:
        print("*********** Save the mean rgb sequences for file {0}".format(file))
        json.dump(mean_rgb_sequence, outfile)
# construct_HCI_dataset_for_ica_method()

def construct_HCI_dataset_for_1dcnn_model(plot=True):
  '''
  Input files:
    mean_rgb_sequence.json (Generated after running the function get_mean_rgb_sequence())
    peak_timestamps_groundtruth_hci_tagging.json (Generated after running preprocess-for-hci-tagging-db.py)
    instan_hr_groundtruth_hci_tagging.json (Generated after running preprocess-for-hci-tagging-db.py)
  Function:
    Detrend and bandpass filter the mean RGB signals as input features: window_length * 3
    Apply sliding window strategy to create the dataset: (window_length * 3, 1)
  Output file:
    processed_windows_hci_tagging.json
  '''
  sequence_file = './mean_rgb_sequence.json'
  groundtruth_file = './peak_timestamps_groundtruth_hci_tagging.json'
  instant_hr_file = './instan_hr_groundtruth_hci_tagging.json'
  with open(sequence_file, 'r') as file:
    sequences = json.load(file)
  with open(groundtruth_file, 'r') as file:
    hci_ground_truth = json.load(file)
  with open(instant_hr_file, 'r') as file:
    instant_hr = json.load(file)

  all_data = []
  for file_idx in sequences.keys(): #sequences.keys()
    # Obtain the input sequence and filter the sequence
    sequence = sequences[file_idx]
    print('Processed vidoe length: {0} minutes'.format(len(sequence) / (FPS * 60)))
    sequence = np.array(sequence)

    Lambda = 300
    window = 23
    order = 128
    b = build_bandpass_filter(FPS, order, False)

    padlen = 3 * len(b)
    if sequence.shape[0] < padlen:
      print("Skipping file (unable to bandpass filter it, the signal is probably not long enough)")
      continue

    rgb_detrend = detrend(sequence, Lambda)
    rgb_averaged = average(rgb_detrend, window)
    from scipy.signal import filtfilt
    rgb_bandpassed = filtfilt(b, np.array([1]), rgb_averaged, axis=0)

    # Plot the result
    if plot:
      sequence = sequence[-500:]
      rgb_detrend = rgb_detrend[-500:]
      rgb_averaged = rgb_averaged[-500:]
      rgb_bandpassed = rgb_bandpassed[-500:]
      # print(sequence.shape, rgb_detrend.shape, rgb_averaged.shape, rgb_bandpassed.shape)

      f, ax = plt.subplots(4, sharex=True)
      ax[0].plot(range(sequence.shape[0]), sequence, 'g')
      ax[0].set_title('Original signal')
      ax[1].plot(range(rgb_detrend.shape[0]), rgb_detrend, 'g')
      ax[1].set_title('After detrending')
      ax[2].plot(range(rgb_averaged.shape[0]), rgb_averaged, 'g')
      ax[2].set_title('After averaging')
      ax[3].plot(range(rgb_bandpassed.shape[0]), rgb_bandpassed, 'g')
      ax[3].set_title('Bandpassed signal')
      plt.show()

      # Obtain peak timestamps as target sequence
      timestamps = hci_ground_truth[file_idx]
      peaks = np.array([0] * (timestamps[-1]+80))
      for ts in timestamps: peaks[ts+80-1] = 1

    # Make the input seq and target seq to have the same length
    timestamps = hci_ground_truth[file_idx]
    instan_hr = instant_hr[file_idx]
    assert len(timestamps) == len(instan_hr)
    total_length = min(len(rgb_bandpassed), timestamps[-1])
    rgb_bandpassed = rgb_bandpassed[: total_length]
    timestamps = [i for i in timestamps if i <= total_length]
    instan_hr = instan_hr[: len(timestamps)]

    # Compute the mean hr seq for every 10 seconds signals from the peak timestamps as groundtruth
    target = []
    for start_point in range(0, total_length - WINDOW_LEN, FPS):
      time_window = [i for i in timestamps if i >= start_point and i < start_point+WINDOW_LEN]
      l_idx = timestamps.index(time_window[0])
      r_idx = timestamps.index(time_window[-1])
      target.append(np.mean(np.array(instan_hr[l_idx: r_idx+1])))

    # Tried to synchronize the input signal and the groundtruth signal
    # for ts in timestamps: peaks[max(0, ts - 1 - 10): min(ts - 1 + 10, len(peaks) - 1)] = 1
    # green_bandpassed = [i[0] for i in rgb_bandpassed]
    # min_green = abs(min(green_bandpassed))
    # green_bandpassed += min_green
    # print(min_green)

    # apply the sliding window strategy to construct the dataset
    sequence = rgb_bandpassed.tolist()
    # target = target.tolist()
    # seq_windows = [[sequence[i: i + WINDOW_LEN], target[i: i + WINDOW_LEN]] for i in range(0, total_length - WINDOW_LEN, FPS)] use BVP peaks as targets
    seq_windows = [[sequence[i: i + WINDOW_LEN], target[int(i/FPS)]] for i in range(0, total_length - WINDOW_LEN, FPS)]
    all_data.extend(seq_windows)
  with open('./processed_windows_hci_tagging.json', 'w') as outfile:
    print("*********** Save {0} windows".format(len(all_data)))
    # print(all_data)
    json.dump({'windows': all_data}, outfile)
# construct_HCI_dataset_for_1dcnn_model()


FACE_PATTERNS = cv2.CascadeClassifier("/Users/qiyinyao/.conda/envs/ica/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml")
def extract_bd_box_from_video(video):
  vc = cv2.VideoCapture(video)
  rois = []
  idx = 0

  try:
    rval, frame = vc.read()
  except:
    raise
  while rval and idx < FPS*60:
    '''Apply a machine learning based approach Haar cascades to detect human face'''
    detector = dlib.get_frontal_face_detector()
    faces = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0)

    # faces = FACE_PATTERNS.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(200, 200))
    if len(faces) == 0 or len(faces) > 1:
      print("{0} faces are detected".format(len(faces)))
      rect = last_rect
    else:
      '''Narrow down the bounding box'''
      rect = faces[0]
    last_rect = rect

    '''Align the bounding box using the predicted facial landmarks'''
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    bd_box = face_utils.FaceAligner(predictor, desiredFaceWidth=256).align(frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), rect) # bd_box.shape = (256, 256, 3)

    '''Crop the bounding box'''
    h, w, c = bd_box.shape
    h_percent = 0.7
    w_percent = 0.7
    cut_x = int(w * (1 - w_percent) * 0.5)
    cut_y = int(h * (1 - h_percent) * 0.5)
    bd_box = bd_box[cut_y: h - cut_y, cut_x: w - cut_x]
    if idx % 100 == 0:
      cv2.imwrite("../results/boxes/" + str(idx) + '.jpg', bd_box) # save the image locally

    rois.append(bd_box.tolist())
    rval, frame = vc.read()
    idx += 1
  return rois

def construct_HCI_dataset_for_3dcnn_model():
  '''
  Input file: mean_hr_groundtruth_hci_tagging.json (Generated after running preprocess-for-hci-tagging-db.py)
  Output file: ./roi_sequence/file_idx.json (Save the bounding box sequences as a json file for each video file)
  '''

  with open('./mean_hr_groundtruth_hci_tagging.json') as json_file:
    file_indices = json.load(json_file).keys()

  dir1 = './Sessions'  # HCI dataset path
  session_files = os.listdir(dir1)
  if '.DS_Store' in session_files:
    session_files.remove('.DS_Store')
  for session in session_files:
    dir2 = dir1 + '/' + session
    files = os.listdir(dir2)
    if '.DS_Store' in files:
      files.remove('.DS_Store')

    idx = 0
    for file in files:
      if file not in file_indices:  # skip files without groundtruth
        print('File {0} does not have the groundtruth data'.format(file))
        continue
      if session + '-' + file + '.json' in os.listdir('../dataset/roi_sequence/'):
        print('File {0} has been processed'.format(file))
        continue
      dir3 = dir2 + '/' + file

      print("Start to extract roi sequence for file:", file)
      sample = os.listdir(dir3)
      video = [ele for ele in sample if ele[-3:] == 'avi'][0]
      roi_sequence = extract_bd_box_from_video(dir3 + '/' + video)
      idx += 1

      with open('./roi_sequence/' + file + '.json', 'w') as outfile:
        print("*********** Save the roi sequence for file {0}".format(file))
        json.dump({file: roi_sequence}, outfile)
# construct_HCI_dataset_for_3dcnn_model()


