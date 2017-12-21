#coding=utf-8
import numpy as np
import struct

def loadImageSet(filename):
	print "load image set", filename
	binfile = open(filename, 'rb')
	buffers = binfile.read()

	head = struct.unpack_from('>IIII', buffers, 0)
	print "head, ", head

	offset = struct.calcsize('>IIII')
	imgNum = head[1]
	width = head[2]
	height = head[3]

	bits = imgNum * width * height
	bitsString = '>' + str(bits) + 'B'

	imgs = struct.unpack_from(bitsString, buffers, offset)
	binfile.close()

	imgs = np.reshape(imgs, [imgNum, width, height, 1])
	print "load imgs finished"
	return imgs

def loadLabelSet(filename):

	print "load label set", filename
	binfile = open(filename, 'rb')
	buffers = binfile.read()

	head = struct.unpack_from('>II', buffers, 0)
	print "head, ", head
	imgNum = head[1]

	offset = struct.calcsize('>II')
	numString = '>' + str(imgNum) + "B"
	labels = struct.unpack_from(numString, buffers, offset)
	binfile.close()

	labels = np.reshape(labels, [imgNum, 1])

	print 'load label finished'
	return labels

def transform(img):
	for i in range(28):
		for j in range(28):
			if img[i,j]>0:
				img[i,j] = 255-img[i,j]
			else:
				img[i,j] = 255

# 255 black, 0 white
