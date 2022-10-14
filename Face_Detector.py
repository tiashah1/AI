# importing libraries
import cv2
from random import randrange

# Load some pre-trained data on face formats from opencv
trained_face_data = cv2.CascadeClassifier("C:/Users/tiash/OneDrive - University of Kent/Desktop/Python projects/Face Detector/haarcascade_frontalface_default.xml")

# To capture video from webcam
webcam = cv2.VideoCapture(0)

# Iterate forever over frames
while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 5)

    # To display the frame with the rectangle around it
    cv2.imshow("Tia's Face Detector", frame)
    key = cv2.waitKey(100)

    # Stop if Q key is pressed
    if key==81 or key==113:
        break


"""
# Choose an image to detect faces in
path = 'C:/Users/tiash/OneDrive - University of Kent/Desktop/Python projects/Face Detector/people.jpg'
img = cv2.imread(path)

# Convert the image to greyscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

#print(face_coordinates)


cv2.imshow("Tia's Face Detector", img)
cv2.waitKey()



print("hello")
"""
