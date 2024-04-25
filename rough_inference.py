import pickle

import cv2
import mediapipe as mp
import numpy as np



default_clear = "default_clear"
default_ok = "default_ok"
default_list = [default_clear, default_ok]
sentence  = "ABC"
my_dict = {}

def clear_sentence():
    global sentence
    sentence = ""
    my_dict.clear()


def add_to_sentence(s):
    
    global sentence
    if(s not in default_list):
        print(s, " not in default_list")
        sentence+= " "+s
    elif 1:
        print(s, " inside default____")
    my_dict.clear()




model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: default_ok}
max_key = "NULL"
while True:

    data_aux = []
    x_ = []
    y_ = []
    ret, frame = cap.read()
    # frame = cv2.imread("0.jpg")

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10


        padding_req = 84 - len(data_aux)
        if(padding_req): 
            data_aux = np.pad(data_aux, (0,padding_req), mode='constant')
        
        
        prediction = model.predict([np.asarray(data_aux)])
        predicted = (prediction[0])
        if predicted in my_dict: 
            my_dict[predicted]+=1
        else:
            my_dict[predicted]=0

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted + " "+str(my_dict[predicted]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3,
                    cv2.LINE_AA)
        
        max_key = max(my_dict, key=my_dict.get)
        if (predicted == default_ok and my_dict[predicted]>5):
            add_to_sentence(max_key)
        if (predicted == default_clear and my_dict[predicted]>5):
            clear_sentence()
        
        
    if(len(my_dict)):
        if(max_key not in default_list):
            cv2.putText(frame, max_key + "  "+ str(my_dict[max_key]), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                cv2.LINE_AA)
        if(my_dict[max_key]>50): my_dict.clear()
    
    cv2.putText(frame, sentence, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
    
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    if cv2.waitKey(25) == ord('q'): break
    print(sentence)


# cap.release()
cv2.destroyAllWindows()
