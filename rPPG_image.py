import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import cv2
from sklearn.decomposition import FastICA

FPS = 30  # frame frequency
timeLength = 20
timeF = 30 / FPS
windowLength = FPS * timeLength
frameNum = 0

MIN_HR_BPM = 40.0
MAX_HR_BMP = 200.0
SEC_PER_MIN = 60
imageNum = 0
def drawPicture(x, y):
    global frameNum
    global imageNum
    plt.figure(figsize=(15, 12))
    font = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 30,
             }
    #plt.title(' channel', font)
    #plt.ylabel(' value', font)
    #plt.xlabel('time/s', font)

    #ticks = [i for i in range(frameNum) if i % FPS == 0]
    #plt.xticks(ticks = ticks, labels = [str(i/FPS) for i in ticks])

    plt.tick_params(labelsize=23)
    plt.plot(x, y, color='red', linewidth=2, linestyle='-')
    #plt.plot(x, y[1], color='green', linewidth=2, linestyle='-')
    #plt.plot(x, y[2], color='blue', linewidth=2, linestyle='-')
    plt.savefig("./result/test" + str(imageNum) + '.jpg')
    plt.show()
    imageNum += 1

def captureVideoFrames():
    global frameNum
    vc = cv2.VideoCapture('./data/hr_80.mp4')
    rval, frame = vc.read()
    c = 0
    face_patterns = cv2.CascadeClassifier(
        "/Users/qiyinyao/anaconda3/envs/minicontest2/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml")
    input = [] # RGB value
    while rval:  # extract frames
        if c % timeF == 0:  # frequency: every timeF frame
            faces = face_patterns.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
            for (x, y, w, h) in faces:
                #print(x, y, w, h)
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #input.append(frame[y+110:y+140, x+25:x+75, :])
                frame = np.array(frame[y:y+h, x:x+w])
                frame = frame[:, :, [2, 1, 0]]  # convert BGR to RGB
                io.imsave("./result/image_ICA/original/" + 'input_' + str(c) + '.png', frame)
                input.append(frame)
            if len(faces) == 0:
                print("One image without detecting human faces")
            cv2.imshow("1", frame)
        if len(input) >= windowLength:
            break
        c = c + 1
        cv2.waitKey()
        rval, frame = vc.read()
    vc.release()
    frameNum = len(input)
    print("Generated frame number:", c / timeF + 1)
    print("Frame number with human face:", frameNum)

    # input shape: windowLength * height * width * 3
    # Applied ICA to extract 2 features for each frame
    # Then average the values over all pixels of the 2 features
    # output shape: windowLength * 2
    output = [[],[]]
    for frame in input:
        h = frame.shape[0]
        w = frame.shape[1]
        frame = frame.reshape((h*w, 3))

        mean = np.mean(frame, axis=0)
        std = np.std(frame, axis=0)
        frame = (frame - mean) / std
        # print("input shape:", frame.shape) # totalPixelNum * 3

        # Apply PCA to extract main features instead of using Whitening
        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=2)
        # input = pca.fit_transform(input)
        # print(input.shape)

        ica = FastICA(n_components= 2, random_state = 0, max_iter = 1000)
        image = ica.fit_transform(frame)
        # image = ica.inverse_transform(image)
        # transformed = transformed.T
        # print("output shape:", image.shape)

        image1 = image[:,0].reshape((h,w))
        image2 = image[:,1].reshape((h,w))

        # plt.savefig("./result/first/"+'first_' + str(c) + '.png')
        # plt.savefig("./result/second/"+'second_' + str(c) + '.png')

        # io.imshow(image1)
        # io.show()

        image1 -= image1.min()
        image1 /= image1.max()

        # image1 *= 255  # [0, 255] range
        # io.imsave("./result/image_ICA/first/" + 'first_' + str(c) + '.png', image1)
        # io.imshow(image2)
        # io.show()

        image2 -= image2.min()
        image2 /= image2.max()
        # image2 *= 255  # [0, 255] range
        # io.imsave("./result/image_ICA/second/" + 'second_' + str(c) + '.png', image2)
        output[0].append(np.mean(image1))
        output[1].append(np.mean(image2))
    output = np.array(output)
    print(output.shape)
    #output = output.T

    powerSpec = np.abs(np.fft.fft(output[0])) ** 2
    freqs = np.fft.fftfreq(windowLength, 1.0 / FPS)
    #maxPwrSrc = np.max(powerSpec, axis=1)
    validIdx = np.where((freqs >= MIN_HR_BPM / SEC_PER_MIN) & (freqs <= MAX_HR_BMP / SEC_PER_MIN))
    validPwr = powerSpec[validIdx]
    validFreqs = freqs[validIdx]
    drawPicture(validFreqs, validPwr)
    maxPwrIdx = np.argmax(validPwr)
    hr = validFreqs[maxPwrIdx] * 60
    print("heartrate:", hr, " times/second")

    powerSpec = np.abs(np.fft.fft(output[1])) ** 2
    freqs = np.fft.fftfreq(windowLength, 1.0 / FPS)
    # maxPwrSrc = np.max(powerSpec, axis=1)
    validIdx = np.where((freqs >= MIN_HR_BPM / SEC_PER_MIN) & (freqs <= MAX_HR_BMP / SEC_PER_MIN))
    validPwr = powerSpec[validIdx]
    validFreqs = freqs[validIdx]
    drawPicture(validFreqs, validPwr)
    maxPwrIdx = np.argmax(validPwr)
    hr = validFreqs[maxPwrIdx] * 60
    print("heartrate:", hr, " times/second")
captureVideoFrames()

# cv2.imshow("1", frame)
# cv2.imwrite("./result/original/" + 'input_' + str(c) + '.png', frame)


# plt.imshow(frame)
# plt.savefig("./result/original/" + 'input_' + str(c) + '.png')
# plt.show()


