# USAGE
# python train.py --dataset dataset

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os
import code

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (224, 224))

def get_images():
    base_path = "multiclass_covid19_dataset/covid-chestxray-dataset/"
    metadata = pd.read_csv(base_path + "metadata.csv")
    label_names = [
        'COVID-19', 'ARDS', 'SARS',
        'Pneumocystis', 'Streptococcus',
        'No Finding', 'Pneumonia'
    ]
    label_encoding = list(range(len(label_names)))
    label_mapping = dict(zip(label_names, label_encoding))
    metadata["finding"] = metadata["finding"].map(label_mapping)
    data = []
    labels = []
    ignored = []
    for index, row in metadata.iterrows():
        try:
            image = preprocess_image(base_path + "images/" + row["filename"])
            data.append(image)
            labels.append(str(row["finding"]))
        except:
            continue
    return labels, data, label_mapping

def prepare_data(data, labels):
    """
    convert the data and labels to NumPy arrays while scaling the pixel
    intensities to the range [0, 255]
    """
    data = np.array(data) / 255.0
    labels = np.array(labels)
    labels = to_categorical(labels)
    return data, labels

def fit_model(trainX, trainY, testX, testY, INIT_LR = 1e-3, EPOCHS = 25):
    """
    load the VGG16 network, ensuring the head FC layer sets are left
    off

    place the head FC model on top of the base model (this will become
    the actual model we will train)
    """
    baseModel = VGG16(weights="imagenet", include_top=False,
                      input_tensor=Input(shape=(224, 224, 3)))

    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(64, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(7, activation="softmax")(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)
    for layer in baseModel.layers:
        layer.trainable = False

    # compile our model
    print("[INFO] compiling model...")
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
    )
    print("[INFO] training head...")
    model_history = model.fit_generator(
        trainAug.flow(trainX, trainY, batch_size=batch_size),
        steps_per_epoch=len(trainX) // batch_size,
        validation_data=(testX, testY),
        validation_steps=len(testX) // batch_size,
        epochs=EPOCHS
    )

    return model

def get_classes(label_encoding):
    label_decoding = dict(
        zip(label_encoding.values(),
            label_encoding.keys()
        )
    )
    classes = []
    for label in testY.argmax(axis=1):
        classes.append(label_decoding[label])
    classes = set(classes)
    return classes

def get_sensitivity_specificity_accuracy(testY, pred_idxs):
    cm = confusion_matrix(testY.argmax(axis=1), pred_idxs)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print(cm)
    print("acc: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))

if __name__ == '__main__':
    plot = "plot.png",
    model_name = "covid19_2.model",
    batch_size = 8
    EPOCHS = 25
    labels, data, label_encoding = get_images()
    data, labels = prepare_data(data, labels)

    trainX, testX, trainY, testY = train_test_split(
        data, labels,
        test_size=0.20,
        stratify=labels,
        random_state=12
    )
    
    # initialize the training data augmentation object
    trainAug = ImageDataGenerator(rotation_range=15,
                                  fill_mode="nearest")

    model = fit_model(trainX, trainY, testX, testY, EPOCHS=EPOCHS)
    print("[INFO] evaluating network...")
    predIdxs = model.predict(testX, batch_size=batch_size)
    predIdxs = np.argmax(predIdxs, axis=1)

    classes = get_classes(label_encoding)
    print(classification_report(testY.argmax(axis=1), predIdxs,
                                target_names=classes))

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, EPOCHS), model_history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, EPOCHS), model_history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, EPOCHS), model_history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, EPOCHS), model_history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on COVID-19 Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plot)
    model.save(model_name, save_format="h5")
