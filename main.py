import numpy as np
import tensorflow as tf
import cv2
cat_model=tf.keras.models.load_model("cat_model.h5")

def image_prediction(img):
    img  = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (250,250))
    img = tf.keras.applications.mobilenet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def predict(img):
    class_names = ['Abyssinian','Bengal','Birman','Bombay','British','Egyptian','Maine','Persian','Ragdoll','Russian''Siamese','Sphynx']
    img = image_prediction(img)
    y_pred = cat_model.predict(img)
    y_pred_label = int(np.argmax(y_pred,axis=1))
    print(f"The cat breed classified by model is {class_names[y_pred_label]}")
path = "Abyssinian_62.jpg"
predict(path)