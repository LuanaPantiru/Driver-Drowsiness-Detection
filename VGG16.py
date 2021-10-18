from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import dlib
import cv2 as cv
import os
import numpy as np
from scipy.spatial import distance

face_detector = dlib.get_frontal_face_detector()


def take_face(faces):
    face = None
    aria_max = 0
    for f in faces:
        l1 = distance.euclidean(np.array([f.left(), f.top()]), np.array([f.left(), f.bottom()]))
        l2 = distance.euclidean(np.array([f.right(), f.bottom()]), np.array([f.left(), f.bottom()]))
        '''l1 = f[2]
        l2 = f[3]'''
        aria = l1 * l2
        if aria >= aria_max:
            aria_max = aria
            face = f
    return face


def frame_capture(name, path, path_alert, path_drowsy, labels):
    '''
    :param name: is the vidoe's name
    :param path: indicates the path of the video
    :param path_alert: the path to the directory where the frames labeled 0 are saved
    :param path_drowsy: the path to the directory where the frames labeled 1 are saved
    :param labels: an 1D array which contains labels for each frame in the video
    :return:
    '''
    video_obj = cv.VideoCapture(path)
    success, frame = video_obj.read()
    fps = int(round(video_obj.get(cv.CAP_PROP_FPS)))
    i = 0
    while success:
        label = int(labels[i])
        faces = face_detector(image, 1)
        if len(faces) != 0:
            if len(faces) > 1:
                face = take_face(faces)
            else:
                face = faces[0]
            if face.left() > 0 and face.top() > 0 and face.right() > 0 and face.bottom() > 0:
                # it will be saved just the face
                frame = frame[face.top():face.bottom(), face.left():face.right()]
                frame = cv.resize(frame, (224, 224))
                frame_flip = cv.flip(frame, 1)
                if label == 0:
                    file_name = os.path.join(path_alert, name + str(i) + '.jpg')
                    cv.imwrite(file_name, frame)
                    file_name = os.path.join(path_alert, name + str(i + 1) + '.jpg')
                    cv.imwrite(file_name, frame_flip)
                else:
                    file_name = os.path.join(path_drowsy, name + str(i) + '.jpg')
                    cv.imwrite(file_name, frame)
                    file_name = os.path.join(path_drowsy, name + str(i + 1) + '.jpg')
                    cv.imwrite(file_name, frame_flip)
        i += fps
        if i >= len(labels):
            break
        video_obj.set(cv.CAP_PROP_POS_FRAMES, i)
        success, frame = video_obj.read()
    video_obj.release()


def read_train(directory_name, path_alert, path_drowsy):
    '''
    :param directory_name:
    :param path_alert: - the path where to save frames with label 0
    :param path_drowsy: - the path where to save frames with label 1
    '''
    # TODO: write code by yourself to save frames for each video from training dataset,
    #  divided them in two directories: alert (frames with label 0) and drowsy (frames with label 1);
    #  use function frame_capture to save all frames for each frame in one video


def read_test(directory_name, path_alert, path_drowsy):
    '''
    :param directory_name:
    :param path_alert: - the path where to save frames with label 0
    :param path_drowsy: - the path where to save frames with label 1
    '''
    # TODO: write code by yourself to save frames for each video from testing dataset,
    #  divided them in two directories: alert (frames with label 0) and drowsy (frames with label 1);
    #  use function frame_capture to save all frames for each frame in one video


# TODO: put your directory name for training dataset
train_file = ...
if len(os.listdir(train_file)) == 0:
    # TODO: put paths where to save frames with label 0 and label 1
    read_train(train_file, ..., ...)

# TODO: put your directory name for testing dataset
test_file = ...
if len(os.listdir(test_file)) == 0:
    # TODO: put paths where to save frames with label 0 and label 1
    read_test(test_file, ..., ...)

trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory=r"trains/images", target_size=(224, 224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory=r"tests/images", target_size=(224, 224))

model = Sequential()
model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(units=4096, activation="relu"))
model.add(Dense(units=4096, activation="relu"))
model.add(Dense(units=2, activation="softmax"))

model.compile(optimizer=optimizers.Adam(learning_rate=0.000001), loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint("vgg16.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='auto')
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=50, verbose=1, mode='auto')
hist = model.fit(traindata, steps_per_epoch=5, epochs=50, validation_data=testdata, validation_steps=1,
                 callbacks=[checkpoint, early])
model.save_weights("vgg16.h5")
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].plot(range(1, len(model.history.history['accuracy']) + 1), model.history.history['accuracy'], linestyle='-',
             marker='o', label='Training Accuracy')
axes[0].plot(range(1, len(model.history.history['val_accuracy']) + 1), model.history.history['val_accuracy'],
             linestyle='-', marker='o', label='Testing Accuracy')
axes[0].set_xlabel('Epochs', fontsize=14)
axes[0].set_ylabel('Accuracy', fontsize=14)
axes[0].set_title('Accuracy Trainig VS Testing', fontsize=14)
axes[0].legend(loc='best')
axes[1].plot(range(1, len(model.history.history['loss']) + 1), model.history.history['loss'], linestyle='solid',
             marker='o', color='crimson', label='Training Loss')
axes[1].plot(range(1, len(model.history.history['val_loss']) + 1), model.history.history['val_loss'], linestyle='solid',
             marker='o', color='dodgerblue', label='Testing Loss')
axes[1].set_xlabel('Epochs', fontsize=14)
axes[1].set_ylabel('Loss', fontsize=14)
axes[1].set_title('Loss Trainig VS Testing', fontsize=14)
axes[1].legend(loc='best')
plt.show()
