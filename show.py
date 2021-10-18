import numpy as np
import cv2 as cv
import dlib
import pickle
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd
import timeit
from sklearn import preprocessing
import matplotlib.animation as animation

predictor = dlib.shape_predictor('E:/data/shape_predictor_68_face_landmarks.dat')
face_detector = dlib.get_frontal_face_detector()


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


def features(landmarks):
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

    A = np.array([landmarks.part(60).x, landmarks.part(60).y])
    B = np.array([landmarks.part(64).x, landmarks.part(64).y])
    E = np.array([landmarks.part(62).x, landmarks.part(62).y])
    F = np.array([landmarks.part(66).x, landmarks.part(66).y])

    mar = distance.euclidean(E, F) / distance.euclidean(A, B)

    moe = mar / (ear + 10 ** (-6))

    return ear, puc, mar, moe


def modify(landmarks):
    new_landmarks = []
    for i in range(36, 68):
        new_landmarks.append(landmarks.part(i).x)
        new_landmarks.append(landmarks.part(i).y)
    new_landmarks = np.array(new_landmarks)
    new_landmarks = new_landmarks.reshape(1, -1)
    return new_landmarks


def show_drowsy(path, model):
    scaler = pickle.load(open('scaler', 'rb'))
    video_obj = cv.VideoCapture(path)
    fps = int(round(video_obj.get(cv.CAP_PROP_FPS)))
    margin = int(fps/3)
    sec = 5
    predictions = []
    count = 0
    video_obj.set(cv.CAP_PROP_POS_FRAMES, count)
    success, frame = video_obj.read()
    pred_anter = 0
    suma_graf = []
    frame_graf = []
    fig = plt.figure()
    graf, = plt.plot(frame_graf, suma_graf)
    plt.ylim((-1, fps * (sec + 1)))
    img = None
    while success:
        # Capture frame-by-frame
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces) > 0:
            if len(faces) > 1:
                faces = take_face(faces)
            for d in faces:
                frame = cv.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 0), 3)
                landmarks = predictor(gray, d)
                landmarks = modify(landmarks)
                landmarks = scaler.transform(landmarks)
                pred = model.predict(landmarks)
                pred_anter = pred
                predictions.append(pred)
        else:
            predictions.append(pred_anter)

        if len(predictions) == margin*sec:
            s = np.sum(predictions)
            suma_graf.append(s*3)
            frame_graf.append(count)
            graf.set_ydata(suma_graf)
            graf.set_xdata(frame_graf)
            plt.xlim((0, count + 1))
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            procent = s * 100 / (margin*sec)
            if procent > 70:
                position = (10, 50)
                frame = cv.putText(frame, "Drowsy", position, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            predictions.pop(0)
            # Display the resulting frame
        if img is not None:
            cv.imshow("plot", img)
        cv.imshow('video', frame)
        if img is None:
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        count += 3
        video_obj.set(cv.CAP_PROP_POS_FRAMES, count)
        success, frame = video_obj.read()
    video_obj.release()
    cv.destroyAllWindows()


"""path = '014_nightnoglasses_mix.mp4'
model = pickle.load(open('svm_model_best', 'rb'))
show_drowsy(path, model)"""