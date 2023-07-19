import os
import random

from playsound import playsound

arr = os.listdir('./suggestion')
print(arr)
r = int(random.random()*12)
print(type(arr))
print(arr[r])
import ctypes
ctypes.windll.user32.MessageBoxW(0, "Your text", "Your title", 1)
playsound('./audio/1.wav')


filename = './audio/Mannipaaya.mp3'
from pydub import AudioSegment
from pydub.playback import play

sound = AudioSegment.from_file(filename, format='mp3')
play(sound)