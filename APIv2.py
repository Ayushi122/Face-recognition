import cv2
import numpy as np
from face_recognition import face_distance, face_encodings, compare_faces, face_locations
#from face_recognition import face_recognition
import os
from PIL import ImageGrab

import logging
#importing libraries for api using flask
from flask import send_file, request, jsonify
from flask import Flask

#For log file
logging.basicConfig(filename='Attendance.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

#importing file for api using flask
app = Flask(__name__)

try:
    @app.route('/Attendance', methods=['POST'])
    def Attendance():
        file = request.files['Employeeimage']
        file.save("input.jpg")

        path = r'C:\Users\Ankit\PycharmProjects\Attendance\ImagesAttendance'
        images = []
        classNames = []
        myList = os.listdir(path)
        print(myList)

        for cl in myList:
            curImg = cv2.imread(f'{path}/{cl}')
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])
        #print(classNames)

        def findEncodings(images):
            encodeList = []
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encode = face_encodings(img)[0]
                encodeList.append(encode)
            return encodeList

        encodeListKnown = findEncodings(images)
        print('Encoding Complete')

        while True:

            img = cv2.imread(r'C:\Users\Ankit\PycharmProjects\Attendance/input.jpg')
            imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_locations(imgS)
            encodesCurFrame = face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = compare_faces(encodeListKnown, encodeFace)
                faceDis = face_distance(encodeListKnown, encodeFace)
                #print(faceDis)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = classNames[matchIndex].lower()
                    print(name)
                    """with open("data.json",'wb') as fr:
                        fr.writelines("Match Found")"""
                    #resp_data = {{'name': name}},{'Match Found'
                    status = {'status':'Match Found'}
                    return status
                else: status = {'status':'Match Not Found'}
                return status
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, a, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            #cv2.imshow('output.jpg', img)
            cv2.imwrite("output.jpg", img)
            cv2.waitKey(0)
        #return send_file(r'C:\Users\Ankit\PycharmProjects\Attendance/output.jpg',attachment_filename='/Attendance/output.jpg')

        return jsonify(status)
except:

   logging.error('Bad Request! Try again!')

if __name__ == '__main__':
    app.run()