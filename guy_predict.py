# gui_predict.py

from tkinter import *
import PIL.ImageGrab as ImageGrab
import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Load trained model
model = load_model("letter_model.h5")

# Convert prediction index to letter
def get_letter(pred_index):
    return chr(pred_index + 64)  # EMNIST labels start from 1 = A (ASCII 65)

# Prediction function
def predict():
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    # Grab the canvas area
    img = ImageGrab.grab().crop((x, y, x1, y1)).convert('L')
    img = img.resize((28, 28))
    img = np.array(img)
    img = np.invert(img)  # Invert to white text on black background
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    pred = model.predict(img)
    label.config(text=f"Predicted Letter: {get_letter(np.argmax(pred))}")

# Clear canvas
def clear():
    canvas.delete("all")
    label.config(text="Draw a letter")

# GUI setup
root = Tk()
root.title("Handwritten Letter Predictor")

canvas = Canvas(root, width=200, height=200, bg="white")
canvas.grid(row=0, column=0, columnspan=2)
canvas.bind("<B1-Motion>", lambda event: canvas.create_oval(event.x, event.y, event.x+10, event.y+10, fill='black'))

Button(root, text="Predict", command=predict).grid(row=1, column=0)
Button(root, text="Clear", command=clear).grid(row=1, column=1)
label = Label(root, text="Draw a letter", font=("Helvetica", 16))
label.grid(row=2, column=0, columnspan=2)

root.mainloop()
