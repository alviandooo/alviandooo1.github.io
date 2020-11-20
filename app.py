from flask import Flask, request,jsonify,send_file,send_from_directory
from werkzeug.utils import secure_filename

from skimage.feature import greycomatrix, greycoprops
from sklearn.preprocessing import StandardScaler  

# import base64
import numpy as np 
import cv2 
import os
import re
# import io
from PIL import Image
import pickle 
import pandas as pd 

from sklearn.preprocessing import LabelEncoder
# from keras.utils.np_utils import to_categorical




app = Flask(__name__, static_url_path='/static')

app.config["IMAGE_UPLOADS"] =  './static'
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]

if __name__ == '__main__':
    app.run()

def getModel():
    global model
    # load the model from disk
    model = pickle.load(open('model_file', 'rb'))
    # result = model.predict(X_test) 
    print(' * model loaded...')

# -------------------- Utility function ------------------------
def normalize_label(str_):
    str_ = str_.replace(" ", "")
    str_ = str_.translate(str_.maketrans("","", "()"))
    str_ = str_.split("_")
    return re.sub(r'\d+$', '',''.join(str_[:2]))

def allowed_image(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit(".",1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else :
        return False

imgs = [] #list image matrix
labels = []

def preprocess_image(image):
    imgs = []
    # img = cv2.imread(image)
    img = cv2.imread(os.path.join(app.config["IMAGE_UPLOADS"], image))  
    # Convert RGB image to grayscale  use cv2.cvtColor
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get Height and Weight from gray shape
    h, w = gray.shape
    # Set ymin, ymax, xmin, xmax from each gray shape
    ymin, ymax, xmin, xmax = h//150, h*149//150, w//150, w*149//150           

    # crop region of interest (ROI) to get important part from citra leaf
    crop = gray[ymin:ymax, xmin:xmax]

    # resize 20% use cv2.resize()
    resize = cv2.resize(crop, (0,0), fx=0.2, fy=0.2)

    imgs.append(resize)
    labels.append(normalize_label(os.path.splitext(image)[0]))

    glcm(imgs,labels)


# ----------------- calculate greycomatrix() & greycoprops() for angle 0, 45, 90, 135 ----------------------------------
def calc_glcm_all_agls(img, label, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    glcm = greycomatrix(img, 
                        distances=dists, 
                        angles=agls, 
                        levels=lvl,
                        symmetric=sym, 
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in greycoprops(glcm, name)[0]]
    for item in glcm_props:
            feature.append(item)
    feature.append(label) 
    
    return feature


# ----------------- call calc_glcm_all_agls() for all properties ----------------------------------
properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

def glcm(imgs, labels):
    glcm_all_agls = []
    for img, label in zip(imgs, labels): 
        glcm_all_agls.append(
                calc_glcm_all_agls(img, 
                                    label, 
                                    props=properties)
                                )
    
    columns = []
    angles = ['0', '45', '90','135']
    for name in properties :
        for ang in angles:
            columns.append(name + "_" + ang)
            
    columns.append("label")

    # Create the pandas DataFrame for GLCM features data
    glcm_df = pd.DataFrame(glcm_all_agls, 
                        columns = columns)
                        
    #save to csv
    # glcm_df.to_csv("1.csv")
    
    knn(glcm_df)

def knn(glcm_df):
    X = glcm_df.iloc[:, 1:-1].values  
    

    # X = glcm_df.iloc[:, 1:-1].values  
    # y = glcm_df.iloc[:, 25].values

    le = LabelEncoder()
    le.fit(glcm_df["label"].values)


    print(" categorical label : \n", le.classes_)

    # Y = le.transform(glcm_df['label'].values)
    # Y = to_categorical(Y)

    # load the scaler from disk
    # scaler = pickle.load(open('scaler_file', 'rb'))

    # X_test_scaled = scaler.transform(X)
     
    # #  # predict test data
    pred = model.predict(X_test_scaled)

    return pred

print(' * Loading K-nn Model...')
getModel()

@app.route('/')
def hello_world():
    return 'hello world'

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        if request.files.get('image'):
            image = request.files['image']
            
            if image.filename == "":
                return jsonify({
                    'status': 'error', 
                    'result': 'filename tidak boleh kosong'
                    }) , 400
            
            if allowed_image(image.filename):
                filename = secure_filename(image.filename)
                image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
                # image = request.files['image'].read()
                # image = Image.open(io.BytesIO(image))
                image = preprocess_image(filename)
                

                return jsonify({
                    'status': 'success', 
                    'result': 'Selesai dong',
                    'score' : pred
                    }) , 400
            else :
                return jsonify({
                    'status': 'error', 
                    'result': 'ups extensi file salah, hanya file jpeg,jpg, png'
                    }) , 400
            

# if __name__ == '__main__':
#     app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT',9000)))             

