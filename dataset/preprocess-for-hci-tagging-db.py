import bob.db.hci_tagging as hci
import json
import os
import cv2

def estimate_heartrate_for_HCI(directory):
    """Estimates the person's heart rate using the ECG sensor data
    Parameters:
      directory (str): A directory name that leads to the location the database
        is installed on the local disk
    """
    from .utils import estimate_heartbeat_peaks, bdf_load_signal

    instan_hr_list = []
    timestamps_list = []
    # choose peaks from ECG3 channel which provides very clean signals
    for channel in ('EXG3', ): # 'EXG1', 'EXG2'
      signal, freq = bdf_load_signal(directory, channel)
      avg_hr, peaks, instan_hr, timestamps = estimate_heartbeat_peaks(signal, freq)
      instan_hr_list.append(instan_hr)
      timestamps_list.append(timestamps)

    # convert from ndarray to python list
    instan_hr = instan_hr_list[0].tolist()
    timestamps = timestamps_list[0].tolist()
    assert len(instan_hr) == len(timestamps)
    return avg_hr, instan_hr, timestamps

def get_hr_groundtruth_json_files():
    dir1 = '.' # HCI dataset path
    session_files = os.listdir(dir1)
    if '.DS_Store' in session_files:
        session_files.remove('.DS_Store')
    mean_hr_groundtruth = {}
    instan_hr_groundtruth = {}
    peak_timestamps_groundtruth = {}
    for session in session_files:
        dir2 = dir1 + '/' + session
        files = os.listdir(dir2)
        if '.DS_Store' in files:
            files.remove('.DS_Store')

        for file in files:
            dir3 = dir2 + '/' + file
            print("For file:", file)
            sample = os.listdir(dir3)
            video = [ele for ele in sample if ele[-3:] == 'avi']
            bdf = [ele for ele in sample if ele[-3:] == 'bdf']
            if len(video) == 0 or len(bdf) == 0:
                continue
            video = video[0]
            bdf = bdf[0]
            avg_hr, instan_hr, timestamps = estimate_heartrate_for_HCI(dir3 + '/' + bdf)
            if avg_hr == -1:
                continue
            mean_hr_groundtruth[file] = avg_hr
            instan_hr_groundtruth[file] = instan_hr
            # print("instantaneous hr sequence:", instan_hr, '\n')

            # construct the reference instantaneous hr values
            vc = cv2.VideoCapture(dir3 + '/' + video)
            fps = int(vc.get(cv2.CAP_PROP_FPS))
            frame_indices = [int(fps * (ts)) for ts in timestamps]
            peak_timestamps_groundtruth[file] = frame_indices
            assert len(instan_hr) == len(frame_indices)

    with open('./mean_hr_groundtruth_hci_tagging.json', 'w') as outfile:
        print("*********** Save the mean hr groundtruth of {0} files".format(len(mean_hr_groundtruth)))
        json.dump(mean_hr_groundtruth, outfile)
    with open('./instan_hr_groundtruth_hci_tagging.json', 'w') as outfile:
        print("*********** Save the instantaneous hr groundtruth of {0} files".format(len(instan_hr_groundtruth)))
        json.dump(instan_hr_groundtruth, outfile)
    with open('./peak_timestamps_groundtruth_hci_tagging.json', 'w') as outfile:
        print("*********** Save the peak timestamps groundtruth of {0} files".format(len(peak_timestamps_groundtruth)))
        json.dump(peak_timestamps_groundtruth, outfile)
get_hr_groundtruth_json_files()