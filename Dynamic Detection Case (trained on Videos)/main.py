import cv2
import streamlit as st
import numpy as np
import tensorflow
from keras.utils.np_utils import to_categorical
import ConvLSTM as CL
#CL.train_model()
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, GRU, Conv2D , MaxPooling2D
from tensorflow.python.keras import layers, models, Sequential, Input
from sklearn.model_selection import train_test_split
#from tensorflow.python.keras.utils import to_categorical
import mediapipe as mp
import os
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import pandas as pd


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = os.path.join('Data')
no_sequences = 30
sequence_length = 30


actions = np.array(['are', 'before' , 'chicago' , 'concentrate','contract', 'drink', 'fine','go', 'hi', 'how', 'no' , 'non_gesture', 'yes', 'you'])

st.title('Empowering Differtly Abled')

st.markdown(
    """
    <style>
    [data-testid ="stSidebar"][aria-expended=True] > div:first-child{
        width: 20%
    }
    [data-testid ="stSidebar"][aria-expended=false] > div:first-child{
        width: 20%
        margin-left: 20%
    }
    
    
    </style/
    
    
    
    """,
    unsafe_allow_html=True

)
st.sidebar.title("Application Side bar")
st.sidebar.subheader("Button and parameters")
st.sidebar.slider('Confidence of gesture', 0.4, 1.0)

@st.cache
def imageresize(image , w = None , h = None):
    h,w = image.shape[:2]

    if w is None and h is None:
        return image
    if w is None:
        r = w/float(w)
        dim = (int(w*r),h)
    else:
        r = w/float(w)
        dim = (w,int(h*r))
    resize = cv2.resize(image,dim)

    return resize



app_mode = st.sidebar.selectbox('Select mode', ['Video streaming' , 'Input new data' , 'Speech recognition'] )


#image = st.sidebar.file_uploader("upload file", ["jpeg" , "jpg" , "png"])
#st.image(image)


#actions = np.array([ 'before' , 'chicago' , 'concentrate','contract', 'drink', 'fine','go', 'no' , 'non_gesture', 'yes',])

def data_read_from_WLASL():
    d = pd.read_csv("D:\WLASL\WLASL_Recognition\\features_df.csv")
    counts = d['gloss'].value_counts()
    counts_df = counts.to_frame().reset_index()
    names = counts_df.loc[counts_df['gloss'] != 3, ['index']]
    names = list(names['index'])
    d.loc[(d['gloss'].isin(names)), 'gloss']=''
    #drop rows that contain specific 'value' in 'column_name'
    data = d[d['gloss'] != '']
    #print(data["gloss"].unique())
    actions = data['gloss'].unique()

    #actions = np.array(['yes'])

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh, face[:-62]])

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results

def take_data():
    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass
    a = st.image("D:\C drive old data\\hi.jpg")
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    image, results = mediapipe_detection(frame, holistic)
                    #                 print(results)

                    #                    # Draw landmarks
                    #
                    #                   # NEW Apply wait logic
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                    (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                    (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    #a.image(image)

                    # NEW Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

        cap.release()
        cv2.destroyAllWindows()
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245),(16, 117, 245),(16, 117, 245),(16, 117, 245),(16, 117, 245),(16, 117, 245),(16, 117, 245),(16, 117, 245),(16, 117, 245),(16, 117, 245),(16, 117, 245),(16, 117, 245),(16, 117, 245),(16, 117, 245)]
def prob_viz(res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                        cv2.LINE_AA)

        return output_frame

def train():
    label_map = {label: num for num, label in enumerate(actions)}
    print(label_map)
    sequences, labels = [], []

    for action in actions:
        for sequence in range(30):
            window = []
            for frame_num in range(30):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1)

    model = Sequential()
    model.add(GRU(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))

    model.add(GRU(128, return_sequences=True, activation='relu'))
    model.add(GRU(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    print("before fit")
    model.fit(X_train, y_train, epochs=500, validation_split=0.2, shuffle=True, batch_size=30)
    print("fit done")
    #model = tensorflow.keras.models.load_model('./newmodel.pkl')
    model.save("./lastestmodel.pkl")


    yhat = model.predict(X_test)

    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()

    multilabel_confusion_matrix(ytrue, yhat)

    print("accuracy " + str(accuracy_score(ytrue, yhat)))
    return model

def train_model2():
    label_map = {label: num for num, label in enumerate(actions)}
    #print(label_map)
    gesture, labels = [], []


    for action in actions:
        print(action)
        for sequence in range(30):
            window = []
            for frame_num in range(30):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}..npy".format(frame_num)))
                res = res.reshape(40, 40, 1)
                window.append(res)
            gesture.append(window)
            labels.append(label_map[action])

    X = np.array(gesture)
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    print("before fit")
    model = Sequential(
        [
        Input(shape=(30, 40, 40, 1)),
        layers.ConvLSTM2D(filters=40, kernel_size=(3,3), padding='same' , return_sequences=False, data_format='channels_last'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(units=14, activation='Softmax')
        ]
    )
# flatten
    #3 dense layers
    #train

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    print("before fit")
    model.fit(X_train, y_train, epochs=100, validation_split=0.2 , batch_size=30)
    print("fit done")
    #model = tensorflow.keras.models.load_model('./newmodel.pkl')
    model.save("./convlstm_original")
    yhat = model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    multilabel_confusion_matrix(ytrue, yhat)
    print("accuracy " + str(accuracy_score(ytrue, yhat)))


def test():
    model = tensorflow.keras.models.load_model("convlstm_original")
    sequence = []
    sentence = []
    res = np.zeros(actions.shape)
    threshold = 0.85
    a = st.image("D:\C drive old data\\hi.jpg")
    text = st.text("")
    s = ''
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)

            key = np.array(keypoints)
            k = key.reshape(40,40,1)
            # if landmarks of right hand nd left are zero than don't append the frame in the sequence.
            sequence.append(k)

            if len(sequence) >= 30 and len(sequence) % 60 == 0:
                sequence = sequence[-50:-20]
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            if actions[np.argmax(res)] != 'non_gesture' and actions[np.argmax(res)] != 'yes' and actions[np.argmax(res)] != 'fine' and actions[np.argmax(res)] != 'you' and actions[np.argmax(res)] != 'are':
                                sentence.append(actions[np.argmax(res)])
                    else:
                        if actions[np.argmax(res)] != 'non_gesture' and actions[np.argmax(res)] != 'yes' and actions[np.argmax(res)] != 'fine' and actions[np.argmax(res)] != 'you' and actions[np.argmax(res)] != 'are':
                                sentence.append(actions[np.argmax(res)])
                                text.text(' '.join(sentence))

                if len(sentence) > 5:
                    sentence = sentence[-5:]
                    for i in sentence:
                        s = s + i + ' '
                    #speak(s)
                #image = prob_viz(res, actions, image, colors)
            #image = prob_viz(res, actions, image, colors)

            text.text(' '.join(sentence))
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence) , (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            a.image(image)




            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if app_mode == 'Video streaming':
    test()


#train()
