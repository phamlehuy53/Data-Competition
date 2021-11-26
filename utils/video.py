# python video.py {video_path} {save_dir} {width} {height}
import enum
import cv2
import sys
import os
from datetime import datetime

pre_file = datetime.now().strftime("%Y%m%d%H%M%s")

def extract_video(vid: str, mw:int, mh:int):
	cap = cv2.VideoCapture(vid)
	if (cap.isOpened()== False): 
		print("Error opening video  file")
		return
	
	# Read until video is completed
	ret, frame = cap.read()
	h, w = frame.shape[:2]
	scale = min( mw/w, mh/h)
	nh, nw = int(h*scale), int(w*scale)
	i = 0
	paused = False
	while(cap.isOpened()):
		
		# Capture frame-by-frame
		if not paused:
			ret, frame = cap.read()
		if ret == True:
		
			# Display the resulting frame
			sized_frame = cv2.resize(frame, (nw, nh))
			cv2.imshow('Frame', sized_frame)
		
			# Press Q on keyboard to  exit
			c = cv2.waitKey(25) 

			if c & 0xFF == ord('p'):
				paused = True
			elif c & 0xFF == ord('c'):
				paused = False
			if c & 0xFF == ord('q'):
				break
			
			if c & 0xFF == ord('s'):
				yield frame
				i+=1
		# Break the loop
		else: 
			break

if __name__ == '__main__':

	vid_path = sys.argv[1]
	save_dir = sys.argv[2]
	mw, mh = [ int(x) for x in sys.argv[3:5] ]
	for i, frame in enumerate(extract_video(vid_path, mw, mh)):
		if frame:
			cv2.imwrite(os.path.join(save_dir, f"{pre_file}{i:05d}.jpg"), frame)
	# extract_video(vid_path, mw, mh)