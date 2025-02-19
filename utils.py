import numpy as np
import skimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import scipy.misc as sm
import os.path

# Weighted method or luminosity method

def rgb2gray(rgb):
	r,g,b = rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
	gray = 0.2989*r + 0.5870*g + 0.1140*b
	return gray


def load_data(dir_name='data'):
	'''
	Load images from "imgs" directory
	Images are in JPG and we convert them to gray scale images
	'''

	# simple version for working with CWD
	DIR = 'data'
	lee = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
	
	imgs = []

	for i in range(0,lee):
		img = mpimg.imread("data/"+str(i)+".jpg")
		img = rgb2gray(img)
		imgs.append(img)

	return imgs

def visualize(imgs,format=None,gray=False):
	# plt.figure(figsize=(20,40))
	# for i,img in enumerate(imgs):
	# 	if(img.shape[0] == 3):
	# 		img = img.transpose(1,2,0)
	# 	plt_idx = i+1
	# 	plt.subplot(2,2,plt_idx)
	# 	plt.imshow(img,format)
	# plt.show()
	x = 1

