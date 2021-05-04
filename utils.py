import os
import urllib.request

UPLOAD_FOLDER = './uploaded'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def download_image_from_url(image_url):
    filename = image_url.split('/')[-1]
    urllib.request.urlretrieve(image_url, UPLOAD_FOLDER + '/' + filename)
    return filename


def save_upload_file(file):
    """
    Save file
    :param file: request.files['image']
    """
    filename = file.filename
    if file and allowed_file(filename):
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        return filename
    else:
        return ''