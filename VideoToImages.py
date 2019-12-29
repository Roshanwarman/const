import numpy as np
import cv2
import argparse
import os
import pytube


parser = argparse.ArgumentParser()
parser.add_argument("url", help="youtube link")
# parser.add_argument("vidPath", help="path for video")
parser.add_argument("destination", help="file name for destination folder")
args = parser.parse_args()


video_url = args.url

yt = pytube.YouTube(video_url)
videoName = yt.title

video = yt.streams.filter(file_extension='mp4').first()
video.download() 


cam = cv2.VideoCapture(videoName + '.mp4')

try: 
    if not os.path.exists(args.destination): 
        os.makedirs(args.destination) 
  
except OSError: 
    print ('Error: Creating directory of data') 
  

currentframe = 0

while(True): 
	ret,frame = cam.read() 

	if ret: 
		name = './'+ args.destination + '/frame' + str(currentframe) + '.jpg'
		print ('Creating...' + name) 

		cv2.imwrite(name, frame) 

		currentframe += 1
	else: 
		break

cam.release() 
cv2.destroyAllWindows() 