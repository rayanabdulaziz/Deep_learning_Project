from flask import Flask, render_template, request
import tensorflow as tf
import cv2
import numpy as np

app = Flask("my app")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def is_allowed(fname):
    return fname.split('.')[-1].lower() in ALLOWED_EXTENSIONS


z_model = tf.keras.models.load_model('model')

def predict_img(img_file, model, shape=(224, 224)):
    # requirements:
    # 1 - load model
    # 2 - cv2: 4.1.2
    # 3 - tf: 2.7.0
    # 4 - numpy
    l2c = {0: 'NonDemented', 1: 'MildDemented', 2: 'ModerateDemented', 3: 'VeryMildDemented'}
    img_arr = cv2.imread(img_file)
    resized = cv2.resize(img_arr, shape)
    label = model.predict(np.array([resized])).argmax(axis=1)[0]
    return label, l2c[label]


@app.route('/', methods=['GET', 'POST'])
def index():
    results = (None, None)
    if(request.method == 'POST'):
        if 'file' not in request.files:
            pass
        else:
            file = request.files.get('file')
            if(file.filename.strip() != ''):
                file.save('ray.jpg')
                results = predict_img('ray.jpg', model=z_model)
                print(results)
    return render_template('index.html', results=results)


app.run(debug=True)