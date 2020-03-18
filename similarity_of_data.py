from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import cv2
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import glob
import code

def extract_vector():
    resnet_feature_list = []
    model = ResNet50(weights="imagenet")
    labels = []
    paths = [
        "similarity_data/covid-chestxray-dataset/images/*",
        "similarity_data/chest_xray/train/NORMAL/*",
        "similarity_data/chest_xray/train/PNEUMONIA/*"
    ]
    for path in paths:
        for im_path in glob.glob(path):
            if path == "similarity_data/covid-chestxray-dataset/images/*":
                labels.append("COVID")
            if path == "similarity_data/chest_xray/train/NORMAL/*":
                labels.append("CLEAR")
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
dbscan = DBSCAN()
dbscan.fit(features)
print(len(set(dbscan.labels_)))
print(len(dbscan.labels_[dbscan.labels_ == 0]))
print(len(dbscan.labels_[dbscan.labels_ == -1]))
print("Number of COVID labels", labels.count("COVID"))
print("Number of CLEAR labels", labels.count("CLEAR"))
print("Number of PNEUMONIA labels", labels.count("PNEUMONIA"))
k_means = KMeans(n_clusters=2)
k_means.fit(features)
normal_and_0 = 0
normal_and_1 = 0
pneumonia_and_0 = 0
pneumonia_and_1 = 0
covid_and_0 = 0
covid_and_1 = 0
for index, label in enumerate(k_means.labels_):
    if label == 0 and labels[index] == "CLEAR":
        normal_and_0 += 1
    if label == 1 and labels[index] == "CLEAR":
        normal_and_1 += 1
    if label == 0 and labels[index] == "PNEUMONIA":
        pneumonia_and_0 += 1
    if label == 1 and labels[index] == "PNEUMONIA":
        pneumonia_and_1 += 1
    if label == 0 and labels[index] == "COVID":
        covid_and_0 += 1
    if label == 1 and labels[index] == "COVID":
        covid_and_1 += 1

print("Normal and 0", normal_and_0)
print("Normal and 1", normal_and_1)
print("COVID and 0", covid_and_0)
print("COVID and 1", covid_and_1)
print("Pneumonia and 0", pneumonia_and_0)
print("Pneumonia and 1", pneumonia_and_1)
