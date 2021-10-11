import cv2 as cv
import os
import numpy as np
import pickle
import dlib
from copy import deepcopy
from sklearn import metrics, preprocessing
from sklearn.svm import LinearSVC
from scipy.spatial import distance


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


def calculate_68landmarks(frame):
    faces = face_detector(frame, 1)
    landmarks_points = None
    landmarks_points_flip = None
    if len(faces) > 0:
        if len(faces) > 1:
            '''in the case that face detection overlap, it will be eliminated'''
            faces = take_face(faces)
        for face in faces:
            frame1 = cv.flip(frame, 1)
            h, w = frame1.shape
            left = w - face.right()
            right = left + (face.right() - face.left())
            flip_face = dlib.rectangle(left, face.top(), right, face.bottom())
            face_landmarks = landmark_predictor(frame, face)
            face_landmarks_flip = landmark_predictor(frame1, flip_face)
            landmarks_points = []
            landmarks_points_flip = []
            for j in range(0, 68):
                p = [face_landmarks.part(j).x, face_landmarks.part(j).y]
                landmarks_points.append(p)
                p = [face_landmarks_flip.part(j).x, face_landmarks_flip.part(j).y]
                landmarks_points_flip.append(p)
    return landmarks_points, landmarks_points_flip


def capture_landmarks(path, labels, flip):
    """
    :param path: indicates the path of the video
    :param labels: an 1D array which contains labels for each frame in the video
    :param flip: boolean parameter which indicates if it should to flip or not the frame
    :return: an 68 X numbers_of_frames array which represent the 68 landmarks for each frame and
            1D array which contains labels for each frame it takes in consideration
    """
    labels = open(labels, 'r')
    labels = labels.read()
    video_obj = cv.VideoCapture(path)
    fps = int(round(video_obj.get(cv.CAP_PROP_FPS)))
    count = 3
    video_obj.set(cv.CAP_PROP_POS_FRAMES, count)
    success, frame = video_obj.read()
    landmarks = []
    new_labels = []
    while success:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        label = int(labels[count])
        landmarks_points, landmarks_points_flip = calculate_68landmarks(frame)
        if landmarks_points is not None:
            landmarks.append(landmarks_points)
            new_labels.append(label)
            if flip is True:
                landmarks.append(landmarks_points_flip)
                new_labels.append(label)
        count += fps
        if count > len(labels):
            break
        video_obj.set(cv.CAP_PROP_POS_FRAMES, count)
        success, frame = video_obj.read()
    video_obj.release()
    return landmarks, new_labels


def read_train(directory_name):
    """
    :return: all_landmarks - an 68 X number_frames_from_all_videos array;
            all_labels - an array with all labels from all videos from training dataset"""

    # TODO: write code by yourself to find landmarks for each video from training dataset;
    # use function capture_landmarks to identify all 68 landmarks for each frame in one video
    all_landmarks, all_labels = [], []
    np.save('train.npy', np.array(all_landmarks))
    np.save('train_labels.npy', np.array(all_labels))
    return all_landmarks, all_labels


def read_test(directory_name):
    """
        :return: all_landmarks - an 68 X number_frames_from_all_videos array;
                all_labels - an array with all labels from all videos from testing dataset"""

    # TODO: write code by yourself to find landmarks for each video from testing dataset;
    # use function capture_landmarks to identify all 68 landmarks for each frame in one video
    all_landmarks, all_labels = [], []
    np.save('test.npy', np.array(all_landmarks))
    np.save('test_labels.npy', np.array(all_labels))
    return all_landmarks, all_labels


face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

train_file = 'train.npy'
train_labels_file = 'train_labels.npy'
if os.path.exists(train_file):
    train = np.load(train_file)
    train_labels = np.load(train_labels_file)
else:
    # TODO: put your directory name for training dataset
    train, train_labels = read_train('...')

test_file = 'test.npy'
test_labels_file = 'test_labels.npy'
if os.path.exists(test_file):
    test = np.load(test_file)
    test_labels = np.load(test_labels_file)
else:
    # TODO: put your directory name for testing dataset
    test, test_labels = read_test('...')

scaler = preprocessing.MinMaxScaler()
train = train.reshape((train.shape[0], train.shape[1] * train.shape[2]))
test = test.reshape((test.shape[0], test.shape[1] * test.shape[2]))
scaler.fit(train)
pickle.dump(scaler, open('scaler', 'wb'))
train = scaler.transform(train)
test = scaler.transform(test)

best_accuracy = 0
best_c = 0
best_model = None
Cs = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 10]

for c in Cs:
    print('Train a classifier for c=%f' % c)
    model = LinearSVC(C=c, max_iter=100000)
    model.fit(train, train_labels)
    pred = model.predict(test)
    acc = metrics.accuracy_score(test, pred) * 100
    if acc > best_accuracy:
        best_accuracy = acc
        best_c = c
        best_model = deepcopy(model)

svm_file_name = 'svm_model_best_' + str(best_accuracy)
pickle.dump(best_model, open(svm_file_name, 'wb'))
print('The best accuracy is: %f' % best_accuracy)