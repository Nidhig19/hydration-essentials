import numpy as np
import os
import tensorflow as tf
import pandas as pd
from keras.models import Model
from flask import Flask,app,request,redirect,render_template,url_for, send_from_directory
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
from tensorflow.python.ops.gen_array_ops import Concat
from keras.models import load_model

model=load_model('models/model.h5')
app=Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def showdashboard():
    return render_template('index.html')

@app.route('/classification.html',methods=['GET','POST'])
def classification():
    result=''
    if request.method=='POST':
        f=request.files['image'] 
        basepath=os.path.dirname(__file__)
        #print("current path",basepath)
        filepath=os.path.join(basepath,'uploads',f.filename)
        #print("upload folder is",filepath)
        f.save(filepath)
        
        img = load_img(filepath,target_size=(256,256))
        x=img_to_array(img)
        x=x.reshape((1,256,256,3))
        
        prediction=model.predict(x)
        per=round((np.max(prediction))*100,2)
        classes = np.argmax(prediction,axis=1)
        op=['Full water level','Half water level','Overflowing']
        result=str(op[classes[0]])
        print(result,prediction)
        return redirect(url_for('predict',per=per,file=f.filename,res=result))
    return render_template('classification.html')

@app.route('/predict.html')
def predict():
    file = request.args.get('file')
    res = request.args.get('res')
    per = request.args.get('per')
    file_url = url_for('upload', filename=file)
    return render_template('predict.html',per=per,file=file_url,prediction=res)

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__=='__main__':
    app.run(debug=True)