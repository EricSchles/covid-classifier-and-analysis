from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import glob
import code

def extract_vector():
    resnet_feature_list = []
    model = ResNet50(weights="imagenet")
    labels = []
    paths = [
        "similarity_data/covid-chestxray-dataset/images/*",
        "similarity_data/chest_xray/train/NORMAL/*",
        "similarity_data/chest_xray/train/PNEUMONIA/*",
        "similarity_data/chest_xray/test/NORMAL/*"
    ]
    for path in paths:
        for im_path in glob.glob(path):
            if path == "similarity_data/covid-chestxray-dataset/images/*":
                labels.append("COVID")
            if path == "similarity_data/chest_xray/train/NORMAL/*":
                labels.append("CLEAR TRAIN")
            if path == "similarity_data/chest_xray/test/NORMAL/*":
                labels.append("CLEAR TEST")
            if path == "similarity_data/chest_xray/train/PNEUMONIA/*":
                labels.append("PNEUMONIA")
            im = cv2.imread(im_path)
            im = cv2.resize(im, (224,224))
            img = preprocess_input(np.expand_dims(im.copy(), axis=0))
            resnet_feature = model.predict(img)
            resnet_feature_np = np.array(resnet_feature)
            resnet_feature_list.append(resnet_feature_np.flatten())

    return np.array(resnet_feature_list), labels

features, labels = extract_vector()
X_train = []
y_train = []
X_test = []
y_test = []
for index, label in enumerate(labels):
    if label == "CLEAR TRAIN":
        X_train.append(features[index])
        y_train.append(0)
    if label == "PNEUMONIA":
        X_train.append(features[index])
        y_train.append(1)
    if label == "COVID":
        X_test.append(features[index])
        y_test.append(1)
    if label == "CLEAR TEST":
        X_test.append(features[index])
        y_test.append(0)

logit_clf = LogisticRegression()
logit_clf.fit(X_train, y_train)
y_pred = logit_clf.predict(X_test)
print(classification_report(y_test, y_pred))
        
