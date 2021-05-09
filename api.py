from __future__ import division, print_function
from flask import Flask, request, jsonify
import utils
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import os
from flask import Flask, request

app = Flask(__name__)
# Model saved with Keras model.save()
MODEL_PATH = './model/model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

@app.route('/')
def hello_world():
    return "Hello World!"

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))  # target_size must agree with what the trained model expects!!

    # Preprocessing the image
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    predict = model.predict(images, batch_size=1)
    return predict

@app.route('/predict', methods=['POST', 'GET'])
def classify():
    try:
        if request.method == 'GET':
            # check if the post request has the image part
            if 'image' not in request.files:
                return jsonify({
                    'message': 'No file part'
                }), 400
            file = request.files['image']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                return jsonify({
                    'message': 'No selected file'
                }), 400
            filename = utils.save_upload_file(file)
            return jsonify({
                'method': "GET",
                'filename': filename
            })
        elif request.method == 'POST' and request.args.get('image_url', '') != '':
            image_url = request.args.get('image_url')
            filename = utils.download_image_from_url(image_url)
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploaded', filename)
            predict = model_predict(file_path, model)

            # Arrange the correct return according to the model.
            # In this model 1 is Pneumonia and 0 is Normal.
            str1 = 'Pneumonia'
            str2 = 'Normal'
            if predict[0][0] == 1:
                return jsonify({
                    'method': 'POST',
                    'image_url': image_url,
                    'file_name': filename,
                    'result': str2
                })
            else:
                return jsonify({
                    'method': 'POST',
                    'image_url': image_url,
                    'file_name': filename,
                    'result': str1
                })
        else:
            return jsonify({
                'message': 'Action is not defined!'
            }), 404
    except Exception as e:
        return repr(e), 500


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)