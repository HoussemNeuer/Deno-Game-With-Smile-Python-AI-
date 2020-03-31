# -------------------------- TRAINER FOR ALL THE ALGORITHMS IN FACE RECOGNITION -------------------------------------------
# ---------------------------------- BY LAHIRU DINALANKARA AKA SPIKE ------------------------------------------------------

import os                                               # importing the OS for path
import cv2                                              # importing the OpenCV library
import numpy as np                                      # importing Numpy library
from PIL import Image                                   # importing Image library

EigenFace = cv2.face.EigenFaceRecognizer_create(15)      # creating EIGEN FACE RECOGNISER


path = 'dataset'                                        # path to the photos
def getImageWithID (path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    FaceList = []
    IDs = []
    for imagePath in imagePaths:
        faceImage = Image.open(imagePath).convert('L')  # Open image and convert to gray
        faceImage = faceImage.resize((110,110))         # resize the image so the EIGEN recogniser can be trained
        faceNP = np.array(faceImage, 'uint8')           # convert the image to Numpy array
        ID = int(os.path.split(imagePath)[-1].split('.')[1])    # Retreave the ID of the array
        FaceList.append(faceNP)                         # Append the Numpy Array to the list
        IDs.append(ID)                                  # Append the ID to the IDs list
        cv2.imshow('Training Set', faceNP)              # Show the images in the list
        cv2.waitKey(1)
    return np.array(IDs), FaceList                      # The IDs are converted in to a Numpy array
IDs, FaceList = getImageWithID(path)

# ------------------------------------ TRAING THE RECOGNISER ----------------------------------------
print('TRAINING......')
EigenFace.train(FaceList, IDs)                          # The recongniser is trained using the images
print('EIGEN FACE RECOGNISER COMPLETE...')
EigenFace.write('trainer/trainingDataEigan.xml')
print('FILE SAVED..')

cv2.destroyAllWindows()
