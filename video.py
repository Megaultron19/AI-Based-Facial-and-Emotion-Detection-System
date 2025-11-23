import cv2
import numpy as np
from keras.models import model_from_json

emotions = {0: "Angry",1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 800))
    if not ret:
        break
    detect = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = detect.detectMultiScale(gframe, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (200, 0, 255), 8)
        CROPPED_FRAME = gframe[y:y + h, x:x + w]
        img = np.expand_dims(np.expand_dims(cv2.resize(CROPPED_FRAME, (48, 48)), -1), 0)

        prediction = model.predict(img)
        maxindex = int(np.argmax(prediction))

        cv2.putText(frame, emotions[maxindex], (x+5, y-20), cv2.FONT_ITALIC, 1, (255,255 , 0), 2, cv2.LINE_AA)
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()
