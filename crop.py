import numpy as np
import imutils
from imutils import face_utils
import dlib
import cv2
import os
import shutil


SLASH_TYPE = "/"
BASE_DIR = "." + SLASH_TYPE

def rect_to_bb(rect):
    """
     Convert a dlib Rect to a bounding box.
     
     @param rect - a dlib Rect to convert. 
     
     @return a tuple of the form (x, y, w, h)
    """
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    """
     Converts a shape to a numpy array of 68x2 coordinates.
     
     @param shape - a shape to be converted.
     @param dtype - the data type of the returned array
     
     @return a numpy array of 68
    """

    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
    	coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords



def crop_miraclvc1_dataset():
    """
    Crop images from the MIRACL-VC1 dataset and save them to the 'cropped' directory.

    The function iterates over a list of people, data types, folder enumerations, and instances,
    and crops all the images in the dataset based on these parameters.
    """
    # Create a new cropped directory if it doesn't exist.
    if not os.path.exists('cropped'):
        os.mkdir('cropped')

    # people = ['F01']
    # data_types = ['words']
    # folder_enum = ['01']
    # instances = ['01']

    # people = ['F01', 'F02', 'F04', 'F05', 'F06', 'F07', 'F08']
    # data_types = ['words']
    # folder_enum = ['01','02','03','04','05','06','07','08', '09', '10']
    # instances = ['01','02','03','04','05','06','07','08', '09', '10']

    people = ['F01','F02','F04','F05','F06','F07','F08','F09', 'F10','F11','M01','M02','M04','M07','M08']
    data_types = ['words']
    folder_enum = ['01','02','03','04','05','06','07','08', '09', '10']
    instances = ['01','02','03','04','05','06','07','08', '09', '10']
    
    i = 1
    
    # Crops all the images in the dataset and saves them to the directory
    for person_ID in people:
        # Create a new directory if necessary
        if not os.path.exists('cropped/' + person_ID ):
            os.mkdir('cropped/' + person_ID + '/')
        
        # Crops all the images in the dataset and saves them to the database
        for data_type in data_types:
            # Create cropped directory if not exists
            if not os.path.exists('cropped/' + person_ID + '/' + data_type):
                os.mkdir('cropped/' + person_ID + '/' + data_type)

            # Crops all the images in the folder_enum and instances in the folder_enum
            for phrase_ID in folder_enum:
                # Create the cropped folder if it doesn't exist
                if not os.path.exists('cropped/' + person_ID + '/' + data_type + '/' + phrase_ID):
                    os.mkdir('cropped/' + person_ID + '/' + data_type + '/' + phrase_ID)

                # Crops all the images in the instance_id
                for instance_ID in instances:
                    directory = BASE_DIR + 'dataset/' + person_ID + '/' + data_type + '/' + phrase_ID + '/' + instance_ID + '/'
                    dir_temp = person_ID + '/' + data_type + '/' + phrase_ID + '/' + instance_ID + '/'
                    print(directory)
                    filelist = os.listdir(directory)
                    
                    # Create the cropped directory if it doesn't exist
                    if not os.path.exists('cropped/' + person_ID + '/' + data_type + '/' + phrase_ID + '/' + instance_ID):
                        os.mkdir('cropped/' + person_ID + '/' + data_type + '/' + phrase_ID + '/' + instance_ID)
                        
                    # Crop all images in filelist
                    for img_name in filelist:
                        # Crop the image to the current directory
                        if img_name.startswith('color'):
                            crop_and_save_image(directory + '' + img_name, 'cropped/' + dir_temp + '' + img_name)
    
    print(f'Iteration : {i}')
    i += 1


def crop_and_save_image(img_path, write_img_path):
    """
     Crop the image to the size of 500x500 and detect faces in the grayscale image and save the cropped image to the specified path.
     
     @param img_path - path to the image to be cropped
     @param write_img_path - path to the image to be saved
     
     @return True if everything went fine False if something went wrong
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(BASE_DIR + 'shape_predictor_68_face_landmarks.dat')
    # load the input image, resize it, and convert it to grayscale

    image = cv2.imread(img_path)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    if len(rects) != 1:
        print("No faces detected for " + img_path)
        return
    # print("RECTS", rects)

    # Write the image to a file.
    rect = rects[0]
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    i, j = 48, 68
    # clone = gray.copy()

    (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))        
    roi = gray[y:y+h, x:x+w]
    roi = imutils.resize(roi, width = 250, inter=cv2.INTER_CUBIC)        
    print(write_img_path, cv2.imwrite(write_img_path, roi))


def crop_dataset(base_folder="cropped", dataset_folder="output_frames", allowed_words=None, max_word_instances=40):
    os.makedirs(base_folder, exist_ok=True)

    for word in os.listdir(dataset_folder):
        if not allowed_words or word in allowed_words:
            print(word)
            word_folder = f"{base_folder}/{word}"
            if(os.path.exists(word_folder)):
                continue

            os.mkdir(word_folder)
            
            for word_instance in os.listdir(f"{dataset_folder}/{word}"):
                word_instance_folder = f"{word_folder}/{word_instance}"
                os.makedirs(word_instance_folder, exist_ok=True)
                
                for frame in os.listdir(f"{dataset_folder}/{word}/{word_instance}"):
                    image_path = f"{dataset_folder}/{word}/{word_instance}/{frame}"
                    output_path = f"{word_instance_folder}/{frame}"
                    crop_and_save_image(image_path, output_path)

                if len(os.listdir(word_instance_folder)) < len(word):
                    shutil.rmtree(word_instance_folder)

                if len(os.listdir(word_folder)) >= max_word_instances:
                    break

if __name__ == "__main__":
    # crop_and_save_image("tst_img.jpg", "tst_img_cropped.jpg")
    # crop_miraclvc1_dataset()
    crop_dataset(allowed_words=["PlayStation", "para", "mundo", "mais", "4", "nada", "por", "grandes", "encontrar"])