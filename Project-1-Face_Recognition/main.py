import cv2
import face_recognition
import numpy as np
from datetime import datetime
import csv



video_capture = cv2.VideoCapture(0)
# load the image sample 
bismay_image = face_recognition.load_image_file("C:/Users/anshu/OneDrive/Documents/Python_Programs/Project-1-Face_Recognition/faces/bismay.jpg")
Anshu_image = face_recognition.load_image_file("C:/Users/anshu\OneDrive/Documents/Python_Programs/Project-1-Face_Recognition/faces/Anshu.jpg")
Durga_image = face_recognition.load_image_file("C:/Users/anshu/OneDrive/Documents/Python_Programs/Project-1-Face_Recognition/faces/durga.jpg")
# encode the face
bismay_encodings = face_recognition.face_encodings(bismay_image)[0]
Anshu_encodings = face_recognition.face_encodings(Anshu_image)[0]
Durga_encodings = face_recognition.face_encodings(Durga_image)[0]

known_faces_encodings = [bismay_encodings,Anshu_encodings,Durga_encodings]
known_faces_names = ["Bismay Bibhu Prakash","Anshuman Ray","Durga Prasad Panda"]


# list of students
students = known_faces_names.copy()

# locate the addresses of the face and the encodings 

face_locations = []
face_encodings = []
name = ""

# get the current date and time 

now = datetime.now()
current_date = now.strftime("%y-%m-%d")

# create a csv writer

f = open(f"C:/Users/anshu/OneDrive/Documents/Python_Programs/Project-1-Face_Recognition\Attendance{current_date}.csv", "a", newline="")
lnWriter = csv.writer(f)

while True:
    _, frames = video_capture.read()
    small_frames = cv2.resize(frames, (0,0), fx = 0.25 ,fy = 0.25)
    rgb_small_frames = cv2.cvtColor(small_frames, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frames)
    face_encodings = face_recognition.face_encodings(rgb_small_frames)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_faces_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if (matches[best_match_index]):
            name = known_faces_names[best_match_index]

        if name in known_faces_names:
            font = cv2.FONT_HERSHEY_COMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frames, name + " is present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

        if name in students:
            students.remove(name)
            current_date = now.strftime("%H-%M-%S")
            lnWriter.writerow([name , current_date])


    cv2.imshow("Attendance", frames) 
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()





