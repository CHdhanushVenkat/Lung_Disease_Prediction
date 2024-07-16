from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model(r"C:\Users\chdha\Downloads\best_model.h5")

UPLOAD_FOLDER = r"C:\Users\chdha\OneDrive\Desktop\CSP Project\chest_xray"

# Create the directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
            file.save(file_path)
            img = image.load_img(file_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x /= 255.0
            classes = model.predict(x)
            result = np.argmax(classes[0])
            return render_template('index3.html', result=result)
    return render_template('index3.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
