# Main file for final computer vision project, CS 4501, Spring 2017
# Author of all code in this file except where otherwise noted: Seven Starosta
# TO RUN: python ModelTrainer.py contentimage styleimage iterations
# color preservation done in another file

from keras.applications import VGG19, vgg19
from keras.layers import Input
from keras.preprocessing import image
from keras import backend as K
from random import randint
from vis.optimizer import Optimizer
from vis import losses
from matplotlib import pyplot
import numpy
import sys
import cv2

#Reading parameters
contentdir = sys.argv[1]
styledir= sys.argv[2]
iterations = int(sys.argv[3])

# gram matrix function, to be used in style loss.
# commented out code was original that gave memory problems.
# input is the first element of the output of a style layer, which gives three dimensional tensor
# output is N*N matrix with inner products as entries
# final code to fix memory errors based on: https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py
def gramMat(i):
    #(x,y,N)=i.shape
    #B=[[0 for a in range(N)]for b in range(N)]
    #for a in range(x):
    #    for b in range(y):
    #        for map1 in range(N):
    #            for map2 in range(N):
    #                B[map1][map2]+=(i[a][b][map1]*i[a][b][map2])
    #return B
	A=K.batch_flatten(K.permute_dimensions(i, (2, 0, 1)))
	B=K.transpose(A)
	return K.dot(A,B)

# content is the output of the content image at layer Conv4_2, i is the output of the image generated at the same layer
# gives mean squared difference of output image and content image
def contentLossF(content,i):
	return K.sum(K.square(content-i))

#style parameter already as gram matrix, i is the output from one of the five style layers
# C is divided by normalizing constant equal to 4* size of gram matrix * (size of output of layer ^2)
def styleLossF(style,i):
	(x,y,N)=i.shape
	B=gramMat(i)
	C=K.sum(K.square(B-style))
	C/= (4.*(int(N)**2)*((int(x)*int(y))**2))
	return C

#classes for use with vis.optimizer.py
#pass in variable 'content' as pre-calculated output from content image, same with 'style' as precalculated gram amtrix
class contentLoss(losses.Loss):
	#pass cmodel.input as array. Pass model. Pass already predicted output.
	def __init__(self,content,i):
		super(contentLoss,self).__init__()
		self.name="Content Loss Function"
		self.content=content
		self.i=i
	def build_loss(self):
		return K.sum(K.square(self.content-self.i))

class styleLoss(losses.Loss):
	def __init__(self,style,i):
		super(styleLoss,self).__init__()
		self.name="Style Loss Function"
		self.style=style
		self.i=i
	def build_loss(self):
		(x,y,N)=self.i.shape
		B=gramMat(self.i)
		C=K.sum(K.square(B-self.style))
		C/= (4.*(int(N)**2)*((int(x)*int(y))**2))
		return C

#White noise function written by Brittany Yu
def whiteNoise(rows,cols):
    noise_matrix = numpy.empty(shape=[rows,cols,3], dtype= numpy.uint8)
    for r in range(0,rows):
        for c in range(0,cols):
            noise_matrix[r][c][0] = randint(200,255)
            noise_matrix[r][c][1] = randint(200,255)
            noise_matrix[r][c][2] = randint(200,255)
    return noise_matrix

img = image.load_img(contentdir)
contentImage=image.img_to_array(img)
width = contentImage.shape[0]
height = contentImage.shape[1]

#takes image, already converted to array.
# Since vgg19 trained on BGR images, switch channels, then remove mean pixel values
def preprocess(img):
    img = numpy.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img[0]

def deprocess(img):
    img=img.reshape((width, height, 3))
    img=img.astype('float64')
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img=img[:, :, ::-1]
    img=numpy.clip(img, 0, 255)
    return img

#contentImage=preprocess(contentImage)

noise=whiteNoise(contentImage.shape[0],contentImage.shape[1])

img = image.load_img(styledir)
styleImage=image.img_to_array(img)
#styleImage=preprocess(styleImage)

input_tensor = Input(shape=contentImage.shape)

trainedModel = VGG19(weights='imagenet',include_top= False,input_tensor=input_tensor)
for i in range(4):
    trainedModel.layers.pop()
trainedModel.outputs = [trainedModel.layers[-1].output]
trainedModel.layers[-1].outbound_nodes = []

layer_dict = dict([(layer.name, layer.output) for layer in trainedModel.layers[1:]])

contentModel = VGG19(weights='imagenet',include_top= False,input_tensor=input_tensor)
for i in range(8):
    contentModel.layers.pop()
contentModel.outputs = [contentModel.layers[-1].output]
contentModel.layers[-1].outbound_nodes = []

styleModel1 = VGG19(weights='imagenet',include_top= False,input_tensor=input_tensor)
for i in range(20):
   styleModel1.layers.pop()
styleModel1.outputs = [styleModel1.layers[-1].output]
styleModel1.layers[-1].outbound_nodes = []

styleModel2 = VGG19(weights='imagenet',include_top= False,input_tensor=input_tensor)
for i in range(17):
    styleModel2.layers.pop()
styleModel2.outputs = [styleModel2.layers[-1].output]
styleModel2.layers[-1].outbound_nodes = []

styleModel3 = VGG19(weights='imagenet',include_top= False,input_tensor=input_tensor)
for i in range(14):
    styleModel3.layers.pop()
styleModel3.outputs = [styleModel3.layers[-1].output]
styleModel3.layers[-1].outbound_nodes = []

styleModel4 = VGG19(weights='imagenet',include_top= False,input_tensor=input_tensor)
for i in range(9):
    styleModel4.layers.pop()
styleModel4.outputs = [styleModel4.layers[-1].output]
styleModel4.layers[-1].outbound_nodes = []

#block5_conv1, needt to repalce 8 with appropriate index offset
styleModel5 = VGG19(weights='imagenet',include_top= False,input_tensor=input_tensor)
for i in range(4):
    styleModel5.layers.pop()
styleModel5.outputs = [styleModel5.layers[-1].output]
styleModel5.layers[-1].outbound_nodes = []

print("done loading models")

contentBase = contentModel.predict(numpy.array([contentImage]))[0, :, :, :]
print("loaded content base")
styleBase1 = gramMat(styleModel1.predict(numpy.array([styleImage]))[0])
print("loaded stylebase1")
styleBase2 = gramMat(styleModel2.predict(numpy.array([styleImage]))[0])
print("loaded stylebase2")
styleBase3 = gramMat(styleModel3.predict(numpy.array([styleImage]))[0])
print("loaded stylebase3")
styleBase4 = gramMat(styleModel4.predict(numpy.array([styleImage]))[0])
print("loaded stylebase4")
styleBase5=gramMat(styleModel5.predict(numpy.array([styleImage]))[0])
print("loaded stylebase5")

contentModel=[] 
styleModel1=[]
styleModel2=[]
styleModel3=[]
styleModel4=[]
styleModel5=[]

#A list of loss functions and their associated weights which get passed into the optimizer class
#use small weights to try to reduce the learning rate in optimizer.py
losses = [
    (contentLoss(contentBase,layer_dict["block4_conv2"][0, :, :, :]),.08),
    (styleLoss(styleBase1,layer_dict["block1_conv1"][0, :, :, :]),.02),
    (styleLoss(styleBase2,layer_dict["block2_conv1"][0, :, :, :]),.02),
    (styleLoss(styleBase3,layer_dict["block3_conv1"][0, :, :, :]),.02),
    (styleLoss(styleBase4,layer_dict["block4_conv1"][0, :, :, :]),.02),
    (styleLoss(styleBase5,layer_dict["block5_conv1"][0, :, :, :]),.02)
]

#the following code was written to use another library, using a function instead of a class
#totalloss=K.variable(0.)
#totalloss+=contentLossF(contentBase,layer_dict["block4_conv2"][0, :, :, :])
#print("made content loss")
#totalloss+=250*styleLossF(styleBase1,layer_dict["block1_conv1"][0, :, :, :])
#print("made style loss 1")
#totalloss+=250*styleLossF(styleBase2,layer_dict["block2_conv1"][0, :, :, :])
#print("made style loss 2")
#totalloss+=250*styleLossF(styleBase3,layer_dict["block3_conv1"][0, :, :, :])
#print("made style loss 3")
#totalloss+=250*styleLossF(styleBase4,layer_dict["block4_conv1"][0, :, :, :])
#print("loaded losses. Calculating gradients...")
#grads = K.gradients(totalloss,trainedModel.input)
#print("Calculated gradients...")

print("Losses initialized. Initializing optimizer")
#first argument are the variables which will be changed by the optimizer and which the output will have partial derivatives taken wrt
opt = Optimizer(trainedModel.input,losses)
print("Optimizer Initialized. Starting optimization")    
finaloutput=opt.minimize(seed_img=noise,max_iter=iterations,verbose=True,progress_gif_path='proggif.gif')[0]
#finaloutput=deprocess(finaloutput)
print("finished")
cv2.imshow('window',finaloutput[:, :, ::-1]) 
cv2.waitKey(0)
cv2.destroyAllWindows() 
#pyplot.imshow(finaloutput)
#pyplot.show() 
print("done with showing")
