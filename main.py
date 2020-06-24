import os
import sys
from PyQt5 import QtWidgets
from urllib import request
import bz2

from src.app import App
from src.utils import attach_logger_to_stdout
from config import models_directory_path


def _download_necessary_files():
    if os.path.exists(os.path.join(os.getcwd(), models_directory_path, 'shape_predictor_68_face_landmarks.dat')):
        return True
    # download it
    print(f'A necessary file is missing. Downloading "shape_predictor_68_face_landmarks.dat"...')
    url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    zip_path = os.path.join(os.getcwd(), models_directory_path, 'shape_predictor_68_face_landmarks.dat.bz2')
    try:
        request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f'Could not download necessary files: {str(e)}')
        return False

    print(f'Extracting the file...')
    zipfile = bz2.BZ2File(zip_path) # open the file
    data = zipfile.read() # get the decompressed data
    newfilepath = zip_path[:-4] # assuming the filepath ends with .bz2
    with open(newfilepath, 'wb') as f:
        f.write(data) # write a uncompressed file
        f.close() # save the file
    print (f'Deleting the archive...')
    os.remove(zip_path)
    print ('Success!')
    return True


def main():
    attach_logger_to_stdout()
    if _download_necessary_files() == False:
        return
    q_app = QtWidgets.QApplication(sys.argv)
    app = App()
    app.display_main_menu()
    sys.exit(q_app.exec())


if __name__ == '__main__':
    main()
