import cv2 as cv
import numpy as np
import pandas as pd
from copy import deepcopy
import pickle
import dlib
from sklearn.svm import LinearSVC
from sklearn import metrics, preprocessing
from scipy.spatial import distance
import os


def take_face(faces):
    face = None
    aria_max = 0
    for f in faces:
        l1 = distance.euclidean(np.array([f.left(), f.top()]), np.array([f.left(), f.bottom()]))
        l2 = distance.euclidean(np.array([f.right(), f.bottom()]), np.array([f.left(), f.bottom()]))
        aria = l1 * l2
        if aria >= aria_max:
            aria_max = aria
            face = f
    return np.array([face])


def calculate_features(frame):
    faces = face_detector(frame)
    if len(faces) == 0:
        faces = face_detector(frame, 1)
    if len(faces) == 0:
        return -1, -1, -1, -1
    if len(faces) > 1:
        faces = take_face(faces)
    for face in faces:
        landmarks = landmark_predictor(frame, face)
        # EAR = eye aspect ratio
        # left eye
        p11 = np.array([landmarks.part(36).x, landmarks.part(36).y])
        p21 = np.array([landmarks.part(37).x, landmarks.part(37).y])
        p31 = np.array([landmarks.part(38).x, landmarks.part(38).y])
        p41 = np.array([landmarks.part(39).x, landmarks.part(39).y])
        p51 = np.array([landmarks.part(40).x, landmarks.part(40).y])
        p61 = np.array([landmarks.part(41).x, landmarks.part(41).y])
        ear1 = (distance.euclidean(p21, p61) + distance.euclidean(p31, p51)) / (2 * distance.euclidean(p11, p41))
        # right eye
        p12 = np.array([landmarks.part(42).x, landmarks.part(42).y])
        p22 = np.array([landmarks.part(43).x, landmarks.part(43).y])
        p32 = np.array([landmarks.part(44).x, landmarks.part(44).y])
        p42 = np.array([landmarks.part(45).x, landmarks.part(45).y])
        p52 = np.array([landmarks.part(46).x, landmarks.part(46).y])
        p62 = np.array([landmarks.part(47).x, landmarks.part(47).y])
        ear2 = (distance.euclidean(p22, p62) + distance.euclidean(p32, p52)) / (2 * distance.euclidean(p12, p42))
        ear = (ear1 + ear2) / 2

        # PUC = pupil circularity
        area1 = (distance.euclidean(p21, p51) / 2) ** 2 * np.pi
        perimeter1 = distance.euclidean(p11, p21)
        perimeter1 += distance.euclidean(p21, p31)
        perimeter1 += distance.euclidean(p31, p41)
        perimeter1 += distance.euclidean(p41, p51)
        perimeter1 += distance.euclidean(p51, p61)
        perimeter1 += distance.euclidean(p61, p11)
        puc1 = 4 * np.pi * area1 / (perimeter1 ** 2)

        area2 = (distance.euclidean(p22, p52) / 2) ** 2 * np.pi
        perimeter2 = distance.euclidean(p12, p22)
        perimeter2 += distance.euclidean(p22, p32)
        perimeter2 += distance.euclidean(p32, p42)
        perimeter2 += distance.euclidean(p42, p52)
        perimeter2 += distance.euclidean(p52, p62)
        perimeter2 += distance.euclidean(p62, p12)
        puc2 = 4 * np.pi * area2 / (perimeter2 ** 2)

        puc = (puc1 + puc2) / 2

        # MAR =
        A = np.array([landmarks.part(60).x, landmarks.part(60).y])
        B = np.array([landmarks.part(64).x, landmarks.part(64).y])
        C = np.array([landmarks.part(61).x, landmarks.part(61).y])
        D = np.array([landmarks.part(67).x, landmarks.part(67).y])
        E = np.array([landmarks.part(62).x, landmarks.part(62).y])
        F = np.array([landmarks.part(66).x, landmarks.part(66).y])
        G = np.array([landmarks.part(63).x, landmarks.part(63).y])
        H = np.array([landmarks.part(65).x, landmarks.part(65).y])
        mar = distance.euclidean(E, F) / distance.euclidean(A, B)

        # MOE =
        moe = mar / (ear + 10 ** (-6))

    return ear, puc, mar, moe


def capture_ear_puc_mar_moe(path, labels, flip):
    ears, pucs, mars, moes = [], [], [], []
    new_labels = []
    video_obj = cv.VideoCapture(path)
    fps = int(round(video_obj.get(cv.CAP_PROP_FPS)))
    margin = int(fps / 2)
    i = margin
    video_obj.set(cv.CAP_PROP_POS_FRAMES, i)
    success, frame = video_obj.read()
    while success:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        label = int(labels[i])
        ear, puc, mar, moe = calculate_features(frame)
        if ear != -1:
            ears.append(ear)
            pucs.append(puc)
            mars.append(mar)
            moes.append(moe)
            new_labels.append(label)
        if flip is True:
            frame = cv.flip(frame, 1)
            ear, puc, mar, moe = calculate_features(frame)
            if ear != -1:
                ears.append(ear)
                pucs.append(puc)
                mars.append(mar)
                moes.append(moe)
                new_labels.append(label)
        i += fps
        if i > len(labels):
            break
        video_obj.set(cv.CAP_PROP_POS_FRAMES, i)
        success, frame = video_obj.read()

    video_obj.release()
    return ears, pucs, mars, moes, new_labels


def read_train(directory_name):
    """
    :return: df - is a DataFrame which contains EAR, PUC, MAR, MOE's values for each frame;
    """

    # TODO: write code by yourself to calculate EAR, PUC, MAR, MOE for each video from training dataset;
    #  use function capture_ear_puc_mar_moe to calculate all 4 features for each frame in one video
    #  and save the results in a CSV
    df = pd.DataFrame([], columns=['Frame', 'EAR', 'PUC', 'MAR', 'MOE','Drowsy'])
    df.to_csv('train.csv')
    return df


def read_test(directory_name):
    """
        :return: df - is a DataFrame which contains EAR, PUC, MAR, MOE's values for each frame;
    """

    # TODO: write code by yourself to find landmarks for each video from testing dataset;
    #  use function capture_ear_puc_mar_moe to calculate all 4 features for each frame in one video
    #  and save the results in a CSV
    df = pd.DataFrame([], columns=['Frame', 'EAR', 'PUC', 'MAR', 'MOE', 'Drowsy'])
    df.to_csv('test.csv')
    return df


face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

train_file = 'train1.csv'
if os.path.exists(train_file):
    df_train = pd.read_csv(train_file)
    df_train = df_train.drop(["Unnamed: 0"], axis=1)
else:
    # TODO: put your directory name for training dataset
    df_train = read_train(...)


test_file = 'test.csv'
if os.path.exists(test_file):
    df_test = pd.read_csv(test_file)
    df_test = df_test.drop(["Unnamed: 0"], axis=1)
else:
    # TODO: put your directory name for testing dataset
    df_test = read_test(...)

x_columns = ['EAR', 'PUC', 'MAR', 'MOE']
y_columns = ['Drowsy']

best_accuracy = 0
best_c = 0
best_model = None
Cs = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0]
for c in Cs:
    print('Train a classifier for c=%f' % c)
    model = LinearSVC(C=c, max_iter=200000)
    model.fit(df_train[x_columns], df_train[y_columns].values.ravel())
    pred = model.predict(df_test[x_columns])
    acc = metrics.accuracy_score(df_test[y_columns], pred) * 100
    print(acc)
    if acc > best_accuracy:
        best_accuracy = acc
        best_c = c
        best_model = deepcopy(model)

print('The best accuracy is: %f' % best_accuracy)

svm_file_name = 'svm_model_4features_norm'+str(best_accuracy)
pickle.dump(best_model, open(svm_file_name, 'wb'))