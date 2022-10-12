import cv2

# Our image
img_file = 'C:/Users/tiash/OneDrive - University of Kent/Desktop/Python projects/Car/cars.jpe g'

# Pre-trained car classifier
classifier_file = 'C:/Users/tiash/OneDrive - University of Kent/Desktop/Python projects/Car/car_detector.xml'

# Create opencv image
img = cv2.imread(img_file)

# Convert to grayscale 
black_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# Detect cars
cars = car_tracker.detectMultiScale(black_white)

# Draw rectangles around the cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


# Display the image with the rectangles around cars
cv2.imshow("Car Detection", img)

# Don't autoclose (wait in the code and listen for a key press)
cv2.waitKey()

print("Code complete")