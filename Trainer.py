
# Copy from LAHIRU DINALANKARA AKA SPIKE
import os                                              
import cv2                                              
import numpy as np                                      
from PIL import Image                                   

EigenFace = cv2.face.EigenFaceRecognizer_create(15)      


path = 'dataset'                                        
def getImageWithID (path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    FaceList = []
    IDs = []
    for imagePath in imagePaths:
        faceImage = Image.open(imagePath).convert('L')  
        faceImage = faceImage.resize((110,110))        
        faceNP = np.array(faceImage, 'uint8')          
        ID = int(os.path.split(imagePath)[-1].split('.')[1])    
        FaceList.append(faceNP)                      
        IDs.append(ID)                                 
        cv2.imshow('Training Set', faceNP)
        cv2.waitKey(1)
    return np.array(IDs), FaceList                     
IDs, FaceList = getImageWithID(path)

print('TRAINING......')
EigenFace.train(FaceList, IDs)                         
print('EIGEN FACE RECOGNISER COMPLETE...')
EigenFace.write('trainer/trainingDataEigan.xml')
print('FILE SAVED..')

cv2.destroyAllWindows()
