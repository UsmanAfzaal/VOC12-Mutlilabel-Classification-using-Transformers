from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, precision_score, recall_score
import collections
import cv2
from models import *
import xml.etree.cElementTree as etree
import keras.backend as K
import numpy as np
import os

# Hyperparameters
batch_size = 16
epochs = 2
image_shape = (128, 128)

def data_text_file(lst: list, name: str):
    with open('VOCdevkit/VOC2012/{}.txt'.format(name), 'w') as f:

        for item in range(len(lst)):
            with open(lst[item], 'r') as xmlDoc:
                xmlDocData = xmlDoc.read()
                xmlDocTree = etree.XML(xmlDocData)

            # filename
            for name in xmlDocTree.iter('filename'):
                filename = name.text[:-4]

            # object list
            for objects in xmlDocTree.iter('object'):
                f.write(filename + " " + objects[0].text)
                f.write('\n')

def data_dict(file_name):
    dictionary = collections.defaultdict(list)

    with open(os.path.join("VOCdevkit/VOC2012/", file_name)) as file:
        for line in file.readlines():
            im_path, label = line.split(" ")

            if label != '\n' and im_path != "":
                if '\n' in label:
                    # split file names and labels
                    label = label[0:len(label) - 1]
                    labels.add(label)
                    # key = filename, value = array of labels
                    dictionary[im_path].append(label)
    return dictionary

# One-hot encoding for the dataset
def one_hot(labels):
    encoding = np.zeros(20)
    for i in range(len(labels)):
        index = class_dictionary_inv[labels[i]]
        encoding[index] = 1
    return encoding

def collect_data(dictionary, img_shape=image_shape):
    images = []
    y = []
    for i, (im_path, tags) in enumerate(dictionary.items()):
        try:
            img = os.path.join("VOCdevkit/VOC2012/JPEGImages/", im_path + ".jpg")
            img = cv2.imread(img)
            img = cv2.resize(img, img_shape)
            img = img[..., ::-1]  # BGR to RGB
        except:
            continue
        images.append(img)
        encoding = one_hot(tags)
        y.append(encoding)
    return images, y

def preprocess_datasets(X_train, y_train, X_val, y_val, X_test, y_test):
    #normalization
    X_train = np.array(X_train, dtype="float") / 255.0
    X_test = np.array(X_test, dtype="float") / 255.0
    X_val = np.array(X_val, dtype="float") / 255.0

    #standardization
    X_train_mean = np.mean(X_train, axis=0)
    X_train -= X_train_mean
    X_test -= X_train_mean
    X_val -= X_train_mean

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_val = np.array(y_val)

    return X_train, y_train, X_val, y_val, X_test, y_test

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

if __name__ == "__main__":

    # Importing list of Input Data
    files_names = os.listdir("VOCdevkit/VOC2012/Annotations")
    file_paths = []
    for file_name in files_names:
        file_paths.append(os.path.join("VOCdevkit/VOC2012/Annotations/", file_name))

    training_data_list = file_paths[0:12000]  # ~70% data for training
    validation_data_list = file_paths[12000:14572]  # ~15% data for training
    testing_data_list = file_paths[14572:17125]  # ~15% data for training

    # Can use the following instead for a quicker test implementation
    """training_data_list = file_paths[0:1200]
    validation_data_list = file_paths[1200:1500]
    testing_data_list = file_paths[1500:2000]"""

    # Making text files with lists of train, val, text split
    data_text_file(training_data_list, "train_labels")
    data_text_file(validation_data_list, "val_labels")
    data_text_file(testing_data_list, "test_labels")

    # making dictionary containing labels for train, val and test
    labels = set()
    training_data = data_dict('train_labels.txt')
    testing_data = data_dict('test_labels.txt')
    validation_data = data_dict('val_labels.txt')

    # Assigning Labels to Classes
    class_dictionary = dict()  # Keys= Class Index: Values= Class names
    class_dictionary_inv = dict()  # Keys= Class names: Values= Class Index
    for i, label in enumerate(sorted(labels)):
        class_dictionary[i] = label
        class_dictionary_inv[label] = i

    # Data Collection
    X_train, y_train = collect_data(training_data)
    X_val, y_val = collect_data(validation_data)
    X_test, y_test = collect_data(testing_data)

    # Data Preprocessing
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_datasets(X_train, y_train, X_val, y_val, X_test, y_test)

    # Model Loading
    model = custom_model(image_shape)
    model.summary()

    # Model Training
    model.compile(loss=binary_crossentropy, optimizer=Adam(), metrics=f1_metric)
    history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_val, y_val), epochs=epochs, verbose=1, shuffle=True)

    # Model Evaluation
    threshold = 0.5
    y_pred = model.predict(X_test)  # Gives an array of float for 20 classes
    # Converting floats to 0 or 1
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    print('\nModel Evaluation on Test Data\n')
    print('Precision', precision_score(y_test, y_pred, average='samples', zero_division=0))
    print('Recall: ', recall_score(y_test, y_pred, average='samples'))
    print('F1-score: ', f1_score(y_test, y_pred, average='samples'))









