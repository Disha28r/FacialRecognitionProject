import cv2
import numpy as np
import face_recognition

# importing image
imgElon = face_recognition.load_image_file('ImagesBasic/Elon Musk.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/Bill Gates.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

# detecting face
faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeingface
encodeElon = face_recognition.face_encodings(imgElon)[0]
# print(faceLoc)
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

# comparing faces and finding distance between them - 128 measurements using linearSVM to find if they match or not
results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
#lower the distance better the match
print(results, faceDis)
#displaying on the image
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,20),cv2.FONT_HERSHEY_COMPLEX,0.4,(0,200,255),1)


cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Test',imgTest)
cv2.waitKey(0)