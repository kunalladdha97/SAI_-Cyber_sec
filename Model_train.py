# MODELING
import cv2
import numpy as np
from os import listdir
from os.path import isfile,join


data_path="C:/Users/Hp/PycharmProjects/Face_Recg/Sample_img/"
files=[f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_data,label=[],[]

for i,file in enumerate(files):
    image_path=data_path+files[i]
    images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    Training_data.append(np.asarray(images,dtype=np.uint8))
    label.append(i)

label=np.asarray(label,dtype=np.int32)
model=cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_data),np.asarray(label))
print("Model training done")

# PREDICTION

face_classifier=cv2.CascadeClassifier('C:/Users/Hp/PycharmProjects/Face_Recg/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def face_detector(img,size=0.5):
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face= face_classifier.detectMultiScale(gray,1.3,5)

    if face is():
        return img,[]

    for(x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        Region_of_intrest=img[y:y+h,x:x+w]
        Region_of_intrest=cv2.resize(Region_of_intrest,(600,600))

    return img,Region_of_intrest

cam=cv2.VideoCapture(0)

while True:
    ret,frame=cam.read()
    image,face=face_detector(frame)

    try:
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        result=model.predict(face)


        # lif resut[1]<300:
        confidence=int(100*(1-(result[1])/300))
        display_string=str(confidence)+"% Confidence it is user"


        if(confidence>90):
            cv2.putText(image, "Face Matched", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image,"Unlocked Succesfully", (260, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper',image)
        else:
            cv2.putText(image, "Unable to recognize you,Try again", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)

    except:
        cv2.putText(image, "Face Not Found", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(1)==13:
        break
cam.release()
cv2.destroyAllWindows()






