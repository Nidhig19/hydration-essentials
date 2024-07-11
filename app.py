import numpy as np
import os
import tensorflow as tf
import pandas as pd
from keras.models import Model
from flask import Flask,app,request,render_template
from keras.models import Model
from keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import Concat
from keras.models import load_model

model =load_model('CNN.h5')
app=Flask(__name__)

@app.route('/')

def showdashboard():
    return render_template('index.html')

@app.route('/classification.html',methods=['GET','POST'])
def classification():
    result=''
    if request.method=='POSt':
        f=request.files['image'] 
        basepath=os.path.dirname(__file__)
        #print("current path",basepath)
        filepath=os.path.join(basepath,'uploads',f.filename)
        #print("upload folder is",filepath)
        f.save(filepath)
        
        img = tf.keras.utils.load_img(filepath,target_size=(128,128))
        x=tf.keras.utils.img_to_array(img)
        x=x/255.0 
        x=np.expand_dims(x,axis=0)
        
        prediction=np.argmax(model.predict(img),axis=1)
        op=['Full water level','Half water level','Overflowing']
        result=str(op[prediction[0]])
        print(result)
    return render_template('classification.html',prediction=result)

if __name__=='__main__':
    app.run(debug=True)