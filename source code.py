import cv2

capture = cv2.VideoCapture('carv2.mp4')


car_cascade = cv2.CascadeClassifier('carx.xml')

while True:
    ret, video = capture.read()
    gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)
    for (x, y, w, h) in cars:
        cv2.rectangle(video, (x, y), (x+w, y+h), (0, 0, 0), 3)
    cv2.imshow('sKSama', video )
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
