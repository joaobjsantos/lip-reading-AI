import cv2
import dlib
from imutils import face_utils
import imutils
import numpy as np
from skimage.transform import resize
import imageio
import nn_model

def recognise_word_miraclvc1():
    MAX_WIDTH = 100
    MAX_HEIGHT = 100

    capture = cv2.VideoCapture("test_video/video.mp4")

    frame_n = 0

    while True:
        success, frame = capture.read()

        if success:
            cv2.imwrite(f"test_video/frames/{frame_n}.jpg", frame)
        else:
            break

        frame_n += 1

    capture.release()
    print(frame_n)

    for frame_i in range(frame_n):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        # load the input image, resize it, and convert it to grayscale

        image = cv2.imread(f'test_video/frames/{frame_i}.jpg')
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = detector(gray, 1)
        if len(rects) > 1:
            print( "ERROR: more than one face detected")
        elif len(rects) < 1:
            print( "ERROR: no faces detected")
        else:
            # print("RECTS", rects)
            for (i, rect) in enumerate(rects):
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                name, i, j = 'mouth', 48, 68
                # clone = gray.copy()

                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))        
                roi = gray[y:y+h, x:x+w]
                roi = imutils.resize(roi, width = 250, inter=cv2.INTER_CUBIC)        
                #print('cropped/' + write_img_path)
                cv2.imwrite(f'test_video/cropped/{frame_i}.jpg', roi)

    sequence = []
    for frame_i in range(frame_n):
        image = imageio.imread(f'test_video/cropped/{frame_i}.jpg')
        image = resize(image, (MAX_WIDTH, MAX_HEIGHT))
        image = 255 * image
        # Convert to integer data type pixels.
        image = image.astype(np.uint8)
        sequence.append(image)  

    sequence = np.array(sequence)
    sequence = np.pad(sequence, [(0, 2), (0, 0), (0, 0)], mode='constant')

    model = nn_model.get_nn_model()
    model.load_weights("checkpoints/cp.ckpt")
    pred = model.predict(sequence)

    words = ['Begin', 'Choose', 'Connection', 'Navigation', 'Next', 'Previous', 'Start', 'Stop', 'Hello', 'Web']
    print(words[np.argmax(pred)])



