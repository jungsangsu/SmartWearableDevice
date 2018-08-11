import socket
import cv2
import numpy as np
import pytesseract
import ast
import urllib.request

from keras.models import load_model
from statistics import mode
from PIL import Image
from darkflow.net.build import TFNet
from darkflow.net.utils.datasets import get_labels
from darkflow.net.utils.inference import detect_faces
from darkflow.net.utils.inference import apply_offsets
from darkflow.net.utils.inference import load_detection_model
from darkflow.net.utils.preprocessor import preprocess_input

#naver API
def trans(inputString):
    resultAPIString =''
    encText = urllib.parse.quote(inputString)
    data = "source=en&target=ko&text=" + encText
    url = "https://openapi.naver.com/v1/language/translate"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if (rescode == 200):
        response_body = response.read()
        resultAPIString = response_body.decode('utf-8')
        resultAPIString = ast.literal_eval(resultAPIString)["message"]["result"]["translatedText"]
        return resultAPIString
    else:
        return 'Error'

#socket 수신 버퍼를 읽어서 반환하는 함수
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)

    return buf


'''
1.Tesseract 설정파일 및 YOLO 옵션 설정하는 부분
2. Naver API KEY
3. Emotion 하이퍼 파라미터 및 모델 설정
'''
options = {
    'model' : 'C:/Users/JSS/PycharmProjects/MainServer/cfg/yolo.cfg',
    'load' : 'C:/Users/JSS/PycharmProjects/MainServer/bin/yolov2.weights',
    'threshold' : 0.3,
    'gpu' : 1.0
}

tfnet = TFNet(options)  #객체를 하나 만들어주기 option[Model, weights]값을 넣어줘서
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
client_id = "ZMnQeD_xU3diiFWqj9JC"
client_secret = "QmkCyJRawH"

# 얼굴검출 알고리즘 및 데이터 모델 지정
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# 하이퍼 파라미터 설정
frame_window = 10
emotion_offsets = (20, 40)

# 모델 불러오기
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# 모델 사이즈 지정
emotion_target_size = emotion_classifier.input_shape[1:3]

emotion_window = []


'''
통신에 관련된 절차
'''
#수신에 사용될 내 ip와 내 port번호
TCP_IP = '192.168.0.254'
# TCP_IP = '127.0.0.1'
TCP_PORT = 30000

#TCP소켓 열고 수신 대기
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(True)
print('--------Client 수신대기---------')
conn, addr = s.accept()

while True :

    inputNumber = (conn.recv(1024)).decode('utf-8') #웨어러블 디바이스로 부터 기능번호를 받음
    print('---------입력받은 신호번호 : {}--------- '.format(inputNumber))
    conn.send('ACK1'.encode('utf-8'))

    # String형의 이미지를 수신받아서 이미지로 변환 하고 화면에 출력
    length = recvall(conn, 16)  # 길이 16의 데이터를 먼저 수신하는 것은 여기에 이미지의 길이를 먼저 받아서 이미지를 받을 때 편리하려고 하는 것이다.
    stringData = recvall(conn, int(length))
    data = np.fromstring(stringData, dtype='uint8')
    decimg = cv2.imdecode(data, 1)
    cv2.imwrite('test.jpg', decimg)

    img = cv2.imread('C:/Users/JSS/PycharmProjects/MainServer/darkflow/net/test.jpg')

    if inputNumber =='1':

        result = tfnet.return_predict(img)
        resultString = ''
        tempList = []
        for i in range(len(result)):
            tempList.append(result[i]['label']+'. ')
        tempList = list(set(tempList))

        if len(tempList) !=0:
            resultString = ''.join(tempList)
            # resultString = trans(resultString)
            conn.send(resultString.encode('utf-8'))
        else :
            resultString = 'Please try again, there is no object in front or a recognition error.'
            conn.send(resultString.encode('utf-8'))

        print("보낸 문자 :" + resultString)

    elif inputNumber=='2':
        dst = 'C:/Users/JSS/PycharmProjects/MainServer/darkflow/net/result.png'

        # Gray Scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Denoising
        img = cv2.fastNlMeansDenoising(img, h=10, searchWindowSize=21, templateWindowSize=7)

        # Thresholding
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

        # Erode       흰배경에 검은 글씨일경우 검정 글씨를 더 굵게 해줌(오히려 인식률이 떨어질수있음)
        # kernel = np.ones((3, 3), np.uint8)
        # img2 = cv2.dilate(img,kernel,iterations=1)
        # img2 = cv2.erode(img,kernel,iterations=1)

        cv2.imwrite(dst, img)
        pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
        im = Image.open('result.png')
        text = pytesseract.image_to_string(im, lang='eng')  # lang설정 [kor =한글 , jpn = 일본어, eng = 영어]

        if len(text) < 1:
            text = 'Please try again, there is no text in front or a recognition error.'
            conn.send(text.encode('utf-8'))
        else :
            # text = trans(text)
            conn.send(text.encode('utf-8'))

        print("보낸 문자 :" + text)

    elif inputNumber =='3':
        emotion_text = None

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detect_faces(face_detection, gray_image)

        for face_coordinates in faces:

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue


        if emotion_text == None :
            emotion_text = 'Please try again, Emotion recognition error.'
            conn.send(emotion_text.encode('utf-8'))
        else:
            # text = trans(text)
            conn.send(emotion_text.encode('utf-8'))

        print("보낸 문자 :" + emotion_text)

    else :
        break

s.close()
