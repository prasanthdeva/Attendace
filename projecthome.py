import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path ='dooraccess'
images = []
classNames=[]
mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    curimg=cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def arrivedhome(name):
    with open('arrived.csv','r+') as f:
        mydata=f.readlines()
        namelist=[]
        for line in mydata:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist :
            now =datetime.now()
            dts=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dts},stored')




encodelistknown = findEncodings(images)
print('encoding completed')

cap = cv2.VideoCapture(0)

while True:
    success, img=cap.read()
    ims=cv2.resize(img,(0,0),None,0.25,0.25)
    ims = cv2.cvtColor(ims, cv2.COLOR_BGR2RGB)


    faceCur = face_recognition.face_locations(ims)
    encodecur = face_recognition.face_encodings(ims,faceCur)

    for encodeface,faceLoc in zip(encodecur,faceCur):
        matches=face_recognition.compare_faces(encodelistknown,encodeface)
        faceDis=face_recognition.face_distance(encodelistknown,encodeface)
        print(faceDis)
        matchindex =np.argmin(faceDis)

        if matches[matchindex]:
            name = classNames[matchindex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 =   y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255.0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            arrivedhome(name)

    cv2.imshow('webcam',img)
    cv2.waitKey(1)

