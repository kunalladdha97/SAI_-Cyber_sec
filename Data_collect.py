# DATA COLLECT

import cv2

face_classifier=cv2.CascadeClassifier('C:/Users/Hp/PycharmProjects/Face_Recg/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None
    for(x,y,w,h) in faces:
        crop_faces=img[y:y+h,x:x+w]
    return crop_faces

cam = cv2.VideoCapture(0)
count=0

while(True):
    ret,frame=cam.read()
    if face_extractor(frame) is not None:
        count+=1
        face=cv2.resize(face_extractor(frame),(600,600))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

        file_path='C:/Users/Hp/PycharmProjects/Face_Recg/Sample_img/user'+str(count)+'.jpg'
        cv2.imwrite(file_path,face)

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(236,240,24),2)
        cv2.imshow('Face Crop',face)

    else:
        print("Face Not Found")

    if cv2.waitKey(1)==13 or count==25:
        break

cam.release()
cv2.destroyAllWindows()
print("Sample collection complete")







