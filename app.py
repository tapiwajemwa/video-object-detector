# from warnings import filterwarnings
from flask import Flask, flash, request, redirect, render_template, jsonify
# from werkzeug.utils import secure_filename
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import tensorflow as tf
import cv2
# import numpy as np
import math
import os
import pandas as pd
# import pandas as pd

UPLOAD_FOLDER = './static/images/'
ALLOWED_EXTENSIONS = {'mp4','avi','mkv','wmv','amv'}
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'super secret key'
# Set the maximum file size to 10 MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

items = ['']
model = load_model('models/inception.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template('upload.html', message='File is too large. File limit is 10mb.'), 413


@app.route('/', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
        # check if the post request has the file part
        file = request.files['file']
       
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            if 'file' in request.files:
                save_path = f"static/video/video.{file.filename.rsplit('.', 1)[1].lower()}"
                file.save(save_path)
                save_video_frames(save_path)
                make_df('static/images/train')
                items = get_classes()

                return render_template('main.html', items = items,message='Objects Detected, you can now search for them')
            else:
                pass
    return render_template('main.html')


@app.route('/search', methods=['POST'])
def search():
    try:
        search_input = request.form['text-input']
        if len(search_input) > 0:
            cluster = generate_image(search_input)
            image_data = base64.b64encode(
                cluster.getvalue()).decode('utf-8')

        return render_template('main.html', image=image_data)
    except Exception as e:
        search_input = ''
        return render_template('main.html')
    

def generate_image(search_text: str) -> BytesIO:
    '''This method returns the image created by matplotlib. It combines all the frames into one figure'''

    dataframe = pd.read_csv('models/preds.csv')
    dataframe = dataframe[['Class', 'Source']]
    image_filenames = os.listdir('static/images/train')
    images = []
    for filename in image_filenames:
        image_path = os.path.join('static/images/train', filename)
        image = Image.open(image_path)
        images.append(image)

    needed_images = dataframe[dataframe['Class'] ==
                              search_text]['Source'].unique().tolist()
    # Step 6: Plot the similar images on one figure with subplots
    fig = plt.figure(figsize=(30, 30), frameon=True)
    fig.suptitle(f'Frames which {search_text}', fontsize=18)
    columns = 2
    rows = 3
    for i in range(1, len(needed_images)+1):
        if i == 6:
            break
        fig.add_subplot(rows, columns, i)
        plt.imshow(images[i-1])
        plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0.3)
    fig.savefig('static/images/result.png')
    img_buf = BytesIO()
    fig.savefig(img_buf, format='png')
    return img_buf


def _predict(frame) -> list:
    '''Predicts what is in a frame and then returns the class which each object detected belongs to'''

    img = cv2.imread(frame)
    # Preprocess the image
    img = cv2.resize(img, (299, 299))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    # Run the image through the model to get predictions
    predictions = model.predict(img)
    # Decode the predictions
    decoded_predictions = tf.keras.applications.imagenet_utils.decode_predictions(predictions)
    preds = []
    # Print the top 2 predictions
    for i in range(2):
        preds.append({'Class': decoded_predictions[0][i][1], 'Source': frame})
        print(decoded_predictions[0][i][1])
    return preds


def get_classes() -> list:
    '''Returns a list of classes available from the original'''

    dataframe = pd.read_csv('models/preds.csv')
    dataframe = dataframe[['Class', 'Source']]
    return dataframe['Class'].unique().tolist()


def make_df(file_path):
    '''Creates a dataframe to generate a csv that'll be used to store classes and respective frames'''

    predictions_df = pd.DataFrame(columns=['Class', 'Source'])
    predictions = []
    for filename in os.listdir(file_path):
        pred = []
        print(filename)

        pred = _predict(os.path.join('static/images/train', filename))
        print(type(pred))

        for prediction in pred:
            predictions_df = predictions_df.append(
                prediction, ignore_index=True)
    predictions_df.to_csv('models/preds.csv')


def save_video_frames(file_path):
    '''Exracts frames from videos and then save them'''

    delete_files('static/images/train')
    count = 0
    # capturing the video from the given path
    cap = cv2.VideoCapture(file_path)
    frameRate = cap.get(5)  # frame rate

    while (cap.isOpened()):
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            # storing the frames in a new folder named train_1
            filename = 'static/images/train/' + 'video' + "_frame%d.jpg" % count
            count += 1
            cv2.imwrite(filename, frame)
    cap.release()


def delete_files(dir):
    '''deletes any files in the storage'''

    file_list = os.listdir(dir)

    for file_name in file_list:
        file_path = os.path.join(dir, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print('error removing file')


def get_images(search_text: str) -> dict:
    '''returns dictionary of required files from the stored csv files'''
    
    dataframe = pd.read_csv('models/preds.csv')
    dataframe = dataframe[['Class', 'Source']]
    needed_images = dataframe[dataframe['Class']
                              == search_text]['Source'].unique().tolist()
    return {'images': needed_images}


if __name__ == '__main__':
    app.run(debug=True)
