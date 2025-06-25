# draw_word_predict.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load your trained letter model
model = load_model("my_model.h5")
classes = np.load("classes.npy")

canvas = np.ones((400, 800), dtype='uint8') * 255
drawing = False

def draw(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(canvas, (x, y), 8, (0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow("Draw Word")
cv2.setMouseCallback("Draw Word", draw)

while True:
    cv2.imshow("Draw Word", canvas)
    key = cv2.waitKey(1)

    if key == ord('c'):
        canvas[:] = 255  # clear canvas

    elif key == ord('p'):
        # Preprocess and segment the word into individual letters
        img = 255 - canvas.copy()  # Invert colors: background black
        img = cv2.GaussianBlur(img, (5, 5), 0)
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        letter_boxes = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])  # sort left to right

        predicted_word = ""

        for ctr in letter_boxes:
            x, y, w, h = cv2.boundingRect(ctr)
            if w > 10 and h > 10:  # ignore tiny noise
                letter_img = thresh[y:y+h, x:x+w]
                letter_img = cv2.resize(letter_img, (28, 28))
                letter_img = letter_img.astype("float32") / 255.0
                letter_img = np.expand_dims(letter_img, axis=(0, -1))

                pred = model.predict(letter_img)
                letter = classes[np.argmax(pred)]
                predicted_word += letter

        print(f"Predicted Word: {predicted_word.upper()}")

    elif key == 27:
        break

cv2.destroyAllWindows()
