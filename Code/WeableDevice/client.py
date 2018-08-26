#!/usr/bin/env python


import signal
import buttonshim
import socket
import cv2
import numpy
import os
import sys
import urllib
import urllib.request

# TTS API INFO
client_id = "mu69nuz21c"
client_secret = "pZHIA8b9JzKc6l300Oa3nv91N3VkFKq6ctFOd6bX"
TCP_IP = '220.67.124.124'
# TCP_IP ='127.0.0.1'
TCP_PORT = 30001
sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))
print('TCP Connection Complete')

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
        os.system('mplayer -ao pulse test.mp3')
        print("save TTS file" + soundfile)
    else:
        print("Error Code: " + rescode)


def main(choice):
    menu = choice
    capture = cv2.VideoCapture(0)
    ret, frame = capture.read()
    sock.send(str(menu).encode('utf-8'))
    response = (sock.recv(1024)).decode('utf-8')
    print('{}'.format(response))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = numpy.array(imgencode)
    stringData = data.tostring()
    x = str(len(stringData))
    sock.send((x.ljust(16)).encode())
    sock.send(stringData)
    textresult = (sock.recv(1024)).decode('utf-8')
    print('{}'.format(textresult))
    capture.release()
    TTS(textresult)

@buttonshim.on_press(buttonshim.BUTTON_A)
def button_a(button, pressed):
    main(1)

@buttonshim.on_press(buttonshim.BUTTON_B)
def button_b(button, pressed):
    main(2)

@buttonshim.on_press(buttonshim.BUTTON_C)
def button_c(button, pressed):
    main(3)

@buttonshim.on_press(buttonshim.BUTTON_D)
def button_d(button, pressed):
    textresult = "Not defined"
    TTS(textresult)

@buttonshim.on_press(buttonshim.BUTTON_E)
def button_e(button, pressed):
    textresult = "Not defined"
    TTS(textresult)

signal.pause()