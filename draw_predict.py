import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Load trained model (make sure model.h5 is in the same folder)
model = tf.keras.models.load_model('model.h5')

# Get number of output classes (10 = digits, 26 = A–Z letters)
num_classes = model.output_shape[-1]

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Draw a Digit or Letter")
        self.geometry("300x360")
        self.resizable(False, False)

        self.canvas = tk.Canvas(self, width=280, height=280, bg='white', cursor='cross')
        self.canvas.pack(pady=10)

        self.button_frame = tk.Frame(self)
        self.button_frame.pack()

        tk.Button(self.button_frame, text="Predict", command=self.predict).grid(row=0, column=0, padx=10)
        tk.Button(self.button_frame, text="Clear", command=self.clear).grid(row=0, column=1, padx=10)

        self.label = tk.Label(self, text="Draw a digit (0–9) or letter (A–Z)", font=("Helvetica", 14))
        self.label.pack(pady=10)

        self.canvas.bind("<B1-Motion>", self.draw)
        self.image = Image.new("L", (280, 280), color=255)
        self.draw_obj = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
        self.draw_obj.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.draw_obj.rectangle([0, 0, 280, 280], fill=255)
        self.label.config(text="Draw a digit (0–9) or letter (A–Z)")

    def predict(self):
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28, 1)
        pred = model.predict(img)
        predicted = np.argmax(pred)

        if num_classes == 10:
            text = f"Predicted Digit: {predicted}"
        elif num_classes == 26:
            letter = chr(predicted + 65)  # 65 = 'A'
            text = f"Predicted Letter: {letter}"
        else:
            text = f"Predicted Class: {predicted}"

        self.label.config(text=text)

if __name__ == "__main__":
    app = App()
    app.mainloop()
