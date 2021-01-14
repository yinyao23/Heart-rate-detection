import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.decomposition import FastICA
from scipy import signal
import os
import json

WINDOW_LENGTH = 30  # time/s
MIN_HR_BPM = 40.0
MAX_HR_BMP = 120.0
MAX_ERROR = 10
IMAGE_IDX = 0
DATASET_PATH = './dataset/Sessions'
FACE_PATTERNS = cv2.CascadeClassifier("/Users/qiyinyao/.conda/envs/ica/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml")

def captureVideoFrames(dir, video):
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

def detectHeartRate(fps, video, sequence):
    def visualize(x, y, path):
        global IMAGE_IDX
        plt.figure(figsize=(15, 12))

        plt.tick_params(labelsize=23)
        plt.plot(x, y[0], color='red', linewidth=2, linestyle='-')
        plt.plot(x, y[1], color='green', linewidth=2, linestyle='-')
        plt.plot(x, y[2], color='blue', linewidth=2, linestyle='-')

        plt.savefig(path + "/signal" + str(IMAGE_IDX) + '.jpg')
        plt.close()
        IMAGE_IDX += 1
    frame_num = len(sequence)
    print("Frame number with human face:", frame_num)

    psd_path = "./results/psd/" + video[:video.index('/')]
    folder = os.path.exists(psd_path)
    if not folder:
        os.makedirs(psd_path)

    '''Normalize the RGB value'''
    sequence = np.array(sequence)
    x = [i for i in range(len(sequence))]
    sequence = signal.detrend(sequence, axis=0)
    visualize(x, sequence.T, psd_path)

    mean = np.mean(sequence, axis=0)
    std = np.std(sequence, axis=0)
    sequence = (sequence - mean) / std

    ''' 
    Apply ICA to clear the RGB signals
    input shape：sequenceLength * 3
    output shape: sequenceLength * 3 
    '''
    predictions = []
    n_window_frames = fps*WINDOW_LENGTH
    print("Frame number of sliding window: ", n_window_frames)
    print("Number of sliding windows:", len(sequence)/fps - WINDOW_LENGTH)
    for start_idx in range(0, len(sequence) - n_window_frames, fps):
        window = sequence[start_idx: start_idx + min(n_window_frames, len(sequence))]
        x = [i for i in range(min(n_window_frames, len(sequence)))]
        visualize(x, window.T, psd_path)

        '''Apply ICA method'''
        # print("ICA input shape:", window.shape)
        ica = FastICA(max_iter=2000, tol = 0.1)
        transformed = ica.fit_transform(window)
        # print("output shape after ICA transformation:", transformed.shape)
        visualize(x, transformed.T, psd_path)

        '''Apply FFT method and PSD method'''
        powerSpec = np.abs(np.fft.fft(transformed, axis=0)) ** 2
        maxPwrSrc = np.max(powerSpec, axis=1)
        freqs = np.fft.fftfreq(len(transformed), 1.0 / fps)

        '''Filter the HR signals using the frequency band'''
        valid_idx = np.where((freqs >= MIN_HR_BPM / 60) & (freqs <= MAX_HR_BMP / 60))
        valid_pwr = maxPwrSrc[valid_idx]
        valid_freqs = freqs[valid_idx]

        visualize(valid_freqs, powerSpec[valid_idx].T, psd_path)
        visualize(valid_freqs, valid_pwr.T, psd_path)

        '''Predict the heart rate'''
        max_pwr_idx = np.argmax(valid_pwr)
        predictions.append(valid_freqs[max_pwr_idx] * 60.0)
    return predictions

if __name__ == "__main__":
    def evaluate(predictions, hr):
        '''
        Evaluate by mean error & RMSE
        '''
        errs = []
        outliers = []
        for prediction in predictions:
            if prediction >= hr - MAX_ERROR and prediction <= hr + MAX_ERROR:
                errs.append(abs(hr - prediction))
            else:
                outliers.append(prediction)
        print("************* Evaluation Results *************")
        print("HR errors:", errs)
        print("Outlier measurements:", outliers)
        return errs, outliers

    with open('dataset/mean_hr_groundtruth_hci_tagging.json') as json_file:
        hci_ground_truth = json.load(json_file)

    sessions = os.listdir(DATASET_PATH)
    if '.DS_Store' in sessions:
        sessions.remove('.DS_Store')

    err_list = []
    rmse_list = []
    outlier_list = []
    for session in sessions:
        dir1 = DATASET_PATH + '/' + session
        for file in os.listdir(dir1):
            dir2 = dir1 + '/' + file
            print("For file:", file)
            sample = os.listdir(dir2)
            video = [ele for ele in sample if ele[-3:] == 'avi'][0]
            signs = [ele for ele in sample if ele[-3:] == 'bdf'][0]

            fps, sequence = captureVideoFrames(dir2, video)
            print("sequence:", sequence)

            predictions = detectHeartRate(fps, file + '/' + video, sequence)
            hr_ground_truth = hci_ground_truth[file]
            print("************* Prediction Results *************")
            print(hr_ground_truth, ':', predictions)

            '''Evaluate the prediction results'''
            if predictions and hr_ground_truth:
                errs, outliers = evaluate(predictions, hr_ground_truth)
                if len(errs) == 0:
                    err = 0
                    rmse = 0
                else:
                    err = np.mean(np.array(errs))
                    rmse = np.sqrt(np.mean(np.array([x * x for x in errs])))
                n_outlier = len(outliers)
                outlier_percent = n_outlier / len(predictions)
                print("Evaluation result: err: {0}, rmse: {1}, outlier_percent: {2} \n".format(err, rmse, outlier_percent))

                err_list.append(err)
                rmse_list.append(rmse)
                outlier_list.append(outlier_percent)
            else:
                print("Error: No prediction results! \n")
    print("The final evaluation results for {0} files are:".format(len(err_list)))
    print("err: {0} mse: {1} outlier_percent: {2} \n".format(np.mean(np.array(err_list)), np.mean(np.array(rmse_list)),
                                                             np.mean(np.array(outlier_percent))))