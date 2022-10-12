import cv2

# Face and smile classifiers
face_detector = cv2.CascadeClassifier('C:/Users/tiash/OneDrive - University of Kent/Desktop/Python projects/Smile/haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('C:/Users/tiash/OneDrive - University of Kent/Desktop/Python projects/Smile/haarcascade_smile.xml')
eye_detector = cv2.CascadeClassifier('C:/Users/tiash/OneDrive - University of Kent/Desktop/Python projects/Smile/haarcascade_eye.xml')

# Get webcam feed
webcam = cv2.VideoCapture(0)

# Show the current frame
while True:
    successful_frame_read, frame = webcam.read()
    
    # Break the loop if there is an error
    if not successful_frame_read:
        break
    
    # Convert the frame to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_detector.detectMultiScale(frame_grayscale)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Get the sub-frame using numpy N-dimensional array splicing
        the_face = frame[y:y+h, x:x+w]
        
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, 
        minNeighbors=20)
        for (x_, y_, w_, h_) in smiles:
            cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_), (255, 0, 0), 4)
        
        eyes = eye_detector.detectMultiScale(face_grayscale, scaleFactor=1.05, 
        minNeighbors=30)
        for (x_, y_, w_, h_) in eyes:
            cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_), (0, 0, 255), 4)

        
        # Label the face as smiling
        if len(smiles) > 0:
            cv2.putText(frame, "Smiling", (x, y+h+40), fontScale=3, 
            fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))
    
    # Show the current frame
    cv2.imshow("Smile Detector", frame)
    # Display
    key = cv2.waitKey(100)

    # Stop if Q key is pressed
    if key==81 or key==113:
        break

webcam.release()
cv2.destroyAllWindows()

print("Code complete")