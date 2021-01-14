#!/usr/bin/env python
# encoding: utf-8

"""Utilities for Remote Photo-Plethysmography Benchmarking"""

import os
import numpy
import bob.io.video
import bob.ip.draw
import bob.ip.facedetect

from mne.preprocessing.ecg import qrs_detector


def bdf_load_signal(fn, name='EXG3', start=None, end=None):
  """Loads a signal named ``name`` from the BDF filenamed ``fn``


  Parameters:

    fn (path): The full path to the file to read
    name (str): The name of the channel to read.
    start (int, option): Start time in seconds
    end (int, optional): End time in seconds


  List of physiological channels used (there are more available, but contain no
  meaningful data) on the Mahnob HCI-Tagging database:

    These are the 32 electrodes from the EEG cap (measurements in uV; for full
    positioning details, see the full database description report, available on
    the database website):

      * AF3
      * AF4
      * C3
      * C4
      * CP1
      * CP2
      * CP5
      * CP6
      * Cz
      * F3
      * F4
      * F7
      * F8
      * FC1
      * FC2
      * FC5
      * FC6
      * Fp1
      * Fp2
      * Fz
      * O1
      * O2
      * Oz
      * P3
      * P4
      * P7
      * P8
      * PO3
      * PO4
      * Pz
      * T7
      * T8

    These are ECG sensors (measurements in uV):

    * EXG1: Upper right corner of chest, under clavicle bone
    * EXG2: Upper left corner of chest, under clavicle bone
    * EXG3: Left side of abdomen (very clean)

    Other sensors:

    * GSR1: Galvanic skin response (in Ohm)
    * Resp: Respiration belt (in uV)
    * Status: Status channel containing markers (Boolean)
    * Temp: Skin temperature on the left pinky (Celsius)

  """

  import pyedflib

  if not os.path.exists(fn):  # or the EdfReader will crash the interpreter
    raise IOError("file `%s' does not exist" % fn)

  with pyedflib.EdfReader(fn) as e:

    # get the status information, so we how the video is synchronized
    status_index = e.getSignalLabels().index('Status')
    sample_frequency = e.samplefrequency(status_index)
    status_size = e.samples_in_file(status_index)
    status = numpy.zeros((status_size,), dtype='float64')
    print('Total length of bdf file: {0} minutes with {1} values'.format(status_size / (sample_frequency * 60),
                                                                         status_size))
    e.readsignal(status_index, 0, status_size, status)
    status = status.round().astype('int')
    nz_status = status.nonzero()[0]
    # print(nz_status)

    # because we're interested in the video bits, make sure to get data
    # from that period only
    # video_start = nz_status[0]
    video_start = 0
    video_end = nz_status[-1]

    # retrieve information from this rather chaotic API
    index = e.getSignalLabels().index(name)
    sample_frequency = e.samplefrequency(index)

    video_start_seconds = video_start / sample_frequency

    if start is not None:
      start += video_start_seconds
      start *= sample_frequency
      if start < video_start: start = video_start
      start = int(start)
    else:
      start = video_start

    if end is not None:
      end += video_start_seconds
      end *= sample_frequency
      if end > video_end: end = video_end
      end = int(end)
    else:
      end = video_end
    # print('start:', start/sample_frequency)
    # print('end:', end/sample_frequency)
    print('Total length after filtering: {0} minutes'.format((end - start) / (sample_frequency * 60)))

    # now read the data into a numpy array (read everything)
    container = numpy.zeros((end - start,), dtype='float64')
    e.readsignal(index, start, end - start, container)

    return container, sample_frequency


def estimate_heartbeat_peaks(s, sampling_frequency):
  '''Estimates the average heart rate taking as base the input signal and its
  sampling frequency.

  This method will use the Pam-Tompkins detector available the MNE package to
  clean-up and estimate the heart-beat frequency based on the ECG sensor
  information provided.

  Returns:

    float: The estimated average heart-rate in beats-per-minute

  '''

  peaks = qrs_detector(sampling_frequency, s)
  try:
    print('Length before handling outlier hr values: {0} minutes'.format(peaks[-1]/(sampling_frequency*60)))

    # identify index of outlier peaks which lead to heart rate lower than 40, higher than 120
    instantaneous_rates = (sampling_frequency * 60) / numpy.diff(peaks)
    hr_selector = (instantaneous_rates >= 40) & (instantaneous_rates <= 120)
    peaks_selector = numpy.insert(hr_selector, 0, True)
    # print(selector_peaks, peaks)

    # replace the outlier hr with the successive hr value
    instantaneous_rates = instantaneous_rates[hr_selector]
    time_stamps = peaks[peaks_selector]/sampling_frequency # remove the timestamp of outlier peaks
    time_stamps = time_stamps[1:] # replace the initial hr with the first hr
    assert len(instantaneous_rates) == len(time_stamps)
    print('Length after handling outlier hr values: {0} minutes'.format(time_stamps[-1]/60))
    return float(numpy.nan_to_num(instantaneous_rates.mean())), peaks, instantaneous_rates, time_stamps
  except:
    print('No detected qrs peaks')
    return -1.0, numpy.array([]), numpy.array([]), numpy.array([])



def plot_signal(s, sampling_frequency, channel_name):
  '''Estimates the heart rate taking as base the input signal and its sampling
  frequency, plots QRS peaks discovered on the base signal.

  This method will use the Pam-Tompkins detector available the MNE package to
  clean-up and estimate the heart-beat frequency based on the ECG sensor
  information provided.

  Returns:

    float: The estimated average heart-rate in beats-per-minute

  '''
  import matplotlib.pyplot as plt

  avg, peaks, _, _ = estimate_heartbeat_peaks(s, sampling_frequency)

  ax = plt.gca()
  ax.plot(numpy.arange(0, len(s) / sampling_frequency, 1 / sampling_frequency),
          s, label='Raw signal');
  xmin, xmax, ymin, ymax = plt.axis()
  ax.vlines(peaks / sampling_frequency, ymin, ymax, colors='r', label='P-T QRS detector')
  plt.xlim(0, len(s) / sampling_frequency)
  plt.ylabel('uV')
  plt.xlabel('time (s)')
  plt.title('Channel %s - Average heart-rate = %d bpm' % (channel_name, avg))
  ax.grid(True)
  ax.legend(loc='best', fancybox=True, framealpha=0.5)
  plt.show()

  return avg, peaks


def detrend(signal, Lambda):
  """applies a detrending filter.

  This code is based on the following article "An advanced detrending method with application
  to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.

  Parameters
  ----------
  signal: numpy.ndarray
    The signal where you want to remove the trend.
  Lambda: int
    The smoothing parameter.

  Returns
  -------
  filtered_signal: numpy.ndarray
    The detrended signal.

  """
  signal_length = signal.shape[0]

  # observation matrix
  H = numpy.identity(signal_length)

  # second-order difference matrix
  from scipy.sparse import spdiags
  ones = numpy.ones(signal_length)
  minus_twos = -2 * numpy.ones(signal_length)
  diags_data = numpy.array([ones, minus_twos, ones])
  diags_index = numpy.array([0, 1, 2])
  D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
  filtered_signal = numpy.dot((H - numpy.linalg.inv(H + (Lambda ** 2) * numpy.dot(D.T, D))), signal)
  # print(signal.shape, filtered_signal.shape)
  return filtered_signal


def average(signal, window_size):
  """Moving average filter.

  Parameters
  ----------
  signal: numpy.ndarray
    The signal to filter.
  window_size: int
    The size of the window to compute the average.

  Returns
  -------
  filtered_signal: numpy.ndarray
    The averaged signal.

  """
  from scipy.signal import lfilter
  a = 1.0
  b = numpy.zeros(window_size)
  b += (1.0 / float(window_size))
  filtered_signal = lfilter(b, a, signal)
  return filtered_signal


def build_bandpass_filter(fs, order, min_freq=0.7, max_freq=4.0, plot=False):
  """builds a butterworth bandpass filter.

  Parameters
  ----------
  fs: float
    sampling frequency of the signal (i.e. framerate).
  order: int
    The order of the filter (the higher, the sharper).
  min_freq: int
    The order of the filter (the higher, the sharper).
  order: int
    The order of the filter (the higher, the sharper).
  plot: bool
    Plots the frequency response of the filter.

  Returns
  -------
  b: numpy.ndarray
    The coefficients of the FIR filter.

  """
  # frequency range in Hertz, corresponds to plausible heart-rate values, i.e. [42-240] beats per minute
  min_freq = 0.7
  max_freq = 4.0

  from scipy.signal import firwin
  nyq = fs / 2.0
  numtaps = order + 1
  b = firwin(numtaps, [min_freq / nyq, max_freq / nyq], pass_zero=False)

  # show the frequency response of the filter
  if plot:
    from matplotlib import pyplot
    from scipy.signal import freqz
    w, h = freqz(b)
    fig = pyplot.figure()
    pyplot.title('Bandpass filter frequency response')
    pyplot.plot(w * fs / (2 * numpy.pi), 20 * numpy.log10(abs(h)), 'b')
    pyplot.axvline(x=min_freq, color="red")
    pyplot.axvline(x=max_freq, color="red")
    pyplot.ylabel('Amplitude [dB]', color='b')
    pyplot.xlabel('Frequency [Hz]')
    pyplot.show()

  return b



def capture_video_frames(dir, video):
    """
    Function: average RGB values over the frame sequence
    Input: directory path && video file name
    Output: a sequence of spatial mean values over RGB channels (Shape: total_frame_number[FPS * time length in seconds] * 3)
    """
    print("For video: {0}".format(video))
    vc = cv2.VideoCapture(dir + '/' + video)
    fps = int(vc.get(cv2.CAP_PROP_FPS))      # fps = vc.get(cv2.CAP_PROP_FPS)
    # print("FPS using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    box_path = "./results/boxes/" + video
    folder = os.path.exists(box_path)
    if not folder:
        os.makedirs(box_path)
    roi_path = "./results/roi/" + video
    folder = os.path.exists(roi_path)
    if not folder:
        os.makedirs(roi_path)
    rois = []

    try:
        rval, frame = vc.read()
    except:
        raise
    fg_model = np.zeros((1, 65), dtype="float")
    bg_model = np.zeros((1, 65), dtype="float")
    idx = 0
    last_rect = []
    while rval:
        # print(frame.shape) # width * height, short * long
        '''Apply a machine learning based approach Haar cascades to detect human face'''
        faces = FACE_PATTERNS.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(200, 200))
        if len(faces) == 0 or len(faces) > 1:
            print("{0} faces are detected".format(len(faces)))
            rect = last_rect
        else:
            '''Narrow down the bounding box'''
            rect = faces[0]
        x, y, w, h = rect
        # print(x, y, w, h)
        last_rect = rect
        h_percent = 1
        w_percent = 0.9
        cut_x = int(w * (1-w_percent)*0.5)
        cut_y = int(h * (1-h_percent)*0.5)

        # '''Save the bounding box result of face detection'''
        # bd_box = frame[y + 2*cut_y: y + h, x + cut_x: x + w - cut_x]

        '''Apply GrabCut algorithm to extract the skin area'''
        mask = np.zeros(frame.shape[:2], np.uint8)
        rect = (x+cut_x, y+cut_y, int(w*w_percent), int(h*h_percent))
        (mask, bgModel, fgModel) = cv2.grabCut(frame, mask, rect, bg_model,
                                               fg_model, iterCount=10, mode=cv2.GC_INIT_WITH_RECT)
        output_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
        output_mask = (output_mask * 255).astype("uint8")
        masked_frame = cv2.bitwise_and(frame, frame, mask=output_mask)
        roi = masked_frame[y+cut_y: y+h-cut_y, x+cut_x: x+w-cut_x]

        '''Save the skin area ROI from the bounding box'''
        if not cv2.imwrite(roi_path + "/face" + str(idx) + '.jpg', roi): # save the image locally
            raise Exception("Could not save the ROI area locally")
        rois.append(roi.reshape(-1, 3))
        idx += 1
        rval, frame = vc.read()
    vc.release()

    sequence = []  # Store averaged RGB value for this video clip
    for roi in rois:
        r, g, b = np.mean(roi, axis=0)
        sequence.append([r, g, b])
    print(len(rois), len(sequence))
    return fps, sequence
