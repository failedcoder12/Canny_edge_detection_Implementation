
import utils
import canny_edge_detection as ced
import os
import cv2
import numpy as np
def generate_video(images): 
    image_folder = 'reqdata' # make sure to use your folder 
    video_name = 'mygeneratedvideo.avi'
    # os.chdir("C:\\Desktop\\Canny Edge Detection") 

    frame = np.uint8(cv2.merge((images[0],images[0],images[0])))
  	
  	# setting the frame width, height width 
    # the width, height of first image 
    height, width, layers = frame.shape
    mid = width//2
    video = cv2.VideoWriter(video_name, 0, 20, (width, height))

    base_height = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    for image in images:
    	x = 0
    	for i in range(height-1,0,-1):
    		if(image[i,mid] == 255):
    			x = height - i
    			break
    	if(base_height == 0):
    		base_height = x
    	orig_height = 35.0/base_height
    	orig_height = orig_height*x
    	new_image = cv2.merge((image,image,image))
    	cv2.putText(new_image,str(orig_height-35.0),(30,30),font,1,(255,255,255),2,cv2.LINE_AA)
    	cv2.putText(new_image,".",(mid,height-x),font,1,(255,0,0),2,cv2.LINE_AA)
    	
    	video.write(np.uint8(new_image))

    # Deallocating memories taken for window creation 
    cv2.destroyAllWindows()  
    video.release()  # releasing the video generated 
  


def save_images1(imgs):
	try:
		if not os.path.exists('reqdata'):
			os.makedirs('reqdata')
	except OSError:
		print('Error: Creating directory of reqdata')

	for i,img in enumerate(imgs):
		if(img.shape[0] == 3):
			img = img.transpose(1,2,0)
		plt_idx = i

		name = './reqdata/frame' + str(plt_idx) + '.jpg'
		print('Creating...' + name)

		cv2.imwrite(name,img)


imgs = utils.load_data()

detector = ced.cannyEdgeDetector(imgs, sigma=1.4, kernel_size=5, lowthreshold=0.34, highthreshold=0.50, weak_pixel=100)

imgs_final = detector.detect()

save_images1(imgs_final)

generate_video(imgs_final)