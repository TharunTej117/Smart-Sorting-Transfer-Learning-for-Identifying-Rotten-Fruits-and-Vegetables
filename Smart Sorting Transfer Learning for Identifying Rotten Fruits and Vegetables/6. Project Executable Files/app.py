from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import sys
import threading
import webbrowser

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")

app = Flask(__name__,
            static_folder=os.path.join(base_path, 'static'),
            template_folder=os.path.join(base_path, 'templates'))

model_path = os.path.join(base_path, "smart_sort_model_multiclass.h5")
model = load_model(model_path)
import sys
import os
from tensorflow.keras.models import load_model

base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_path, 'smart_sort_model_multiclass.h5')
model = load_model(model_path)
class_names = [
    'FreshApple', 'FreshMango', 'FreshOrange', 'FreshPotato', 'FreshTomato',
    'RottenApple', 'RottenMango', 'RottenOrange', 'RottenPotato', 'RottenTomato'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None
    if request.method == 'POST':
        img_file = request.files['image']
        if img_file:
            filename = img_file.filename
            save_path = os.path.join(app.static_folder, filename)
            img_file.save(save_path)

            img = image.load_img(save_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            predicted_label = class_names[np.argmax(preds)]
            prediction = f"Prediction: {predicted_label}"
            image_path = f'static/{filename}'

    return render_template('index.html', prediction=prediction, image_path=image_path)

if __name__ == "__main__":
    def open_browser():
        webbrowser.open_new("http://localhost:5000")

    threading.Timer(1.25, open_browser).start()
    app.run(debug=False)