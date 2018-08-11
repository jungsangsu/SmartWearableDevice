import socket
import cv2
import numpy
import os
import sys
import urllib.request
import urllib


def TTS(inputString):
    encText = urllib.parse.quote(inputString)
    data = "speaker=matt&speed=0&text=" + encText
    url = "https://naveropenapi.apigw.ntruss.com/voice/v1/tts"
    request = urllib.request.Request(url)
    request.add_header("X-NCP-APIGW-API-KEY-ID", client_id)
    request.add_header("X-NCP-APIGW-API-KEY", client_secret)
    response = urllib.request.urlopen(request, data=data.encode('utf-8'))
    rescode = response.getcode()
    soundfile = 'test.mp3'
    if (rescode == 200):
        response_body = response.read()
        with open(soundfile, 'wb') as f:
            f.write(response_body)
        os.system('omxplayer test.mp3')
        print("save TTS file" + soundfile)
    else:
        print("Error Code: " + rescode)


# TTS API INFO
client_id = "mu69nuz21c"
client_secret = "pZHIA8b9JzKc6l300Oa3nv91N3VkFKq6ctFOd6bX"

# 연결할 서버(수신단)의 ip주소와 port번호
TCP_IP = '220.67.124.124'
# TCP_IP ='127.0.0.1'
TCP_PORT = 30000

# 송신을 위한 socket 준비
sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))
print('TCP 연결완료')
# OpenCV를 이용해서 webcam으로 부터 이미지 추출


while True:
    meue = input('1.object 2.text 3.Emotion')
    capture = cv2.VideoCapture(0)
    ret, frame = capture.read()  # 사진 한프레임을 읽는 부분

    sock.send(str(meue).encode('utf-8'))  # 옵션을 전송

    response = (sock.recv(1024)).decode('utf-8')
    print('{}'.format(response))

    # 추출한 이미지를 String 형태로 변환(인코딩)시키는 과정
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = numpy.array(imgencode)
    stringData = data.tostring()
    x = str(len(stringData))
    # String 형태로 변환한 이미지를 socket을 통해서 전송
    sock.send((x.ljust(16)).encode())
    sock.send(stringData)

    textresult = (sock.recv(1024)).decode('utf-8')
    print('{}'.format(textresult))
    capture.release()

    # TTS(textresult)

sock.close()
print('연결 끝')