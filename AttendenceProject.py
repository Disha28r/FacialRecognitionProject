import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#creating a list which can import image directly from the folder
path = 'ImagesAttendence'
images = []
classNames = []
#grabbing list of images from the folder
myList = os.listdir(path)
print(myList)

#use names and import image one by one
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    #for removing .jpg
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

#simple function that will compute all the encodings for us
def findEncodings(images):
    encodeList = []
    for img in images:
         img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
         encode = face_recognition.face_encodings(img)[0]
         encodeList.append(encode)
    return encodeList

#marking attendence
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        # print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')






encodeListKnown = findEncodings(images)
# print(len(encodeListKnown))
print('Encoding Complete!')

#test images from  webcam
#to intialize the webcam
cap = cv2.VideoCapture(0)
#while loop to get each frames
while True:
    success, img = cap.read()
#reduce image size
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #in the webcam we can find multiple faces so to find our image location
    faceCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    #finding matches - iterate all matches from current frame and compare with the one we found from inputted image
    for encodeFace,faceLoc in zip(encodesCurFrame,faceCurFrame): #zip for using it in same loop
        #one by one grabs face location from facecurframe list and it will grab encoding of encodeface from encodescurframe
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)


        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1,x2,y2,x1 = faceLoc
            # y1, x2, y2, x1 = y1*4 ,x2*4 ,y2*4 ,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
            markAttendance(name)


    # display bounding box and write name
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)




# faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
#
# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
#
#
# results = face_recognition.compare_faces([encodeElon],encodeTest)
# faceDis = face_recognition.face_distance([encodeElon],encodeTest)