from keras.applications import VGG19
from keras.layers import Input
from keras.preprocessing import image
from keras import backend as K
from random import randint
from vis.optimizer import Optimizer
from vis import losses
from matplotlib import pyplot
import numpy
#import cv2

def gramMat(input):
    #(x,y,N)=A.shape
    #B=[[0 for a in range(N)]for b in range(N)]
    #for a in range(x):
    #    for b in range(y):
    #        for map1 in range(N):
    #            for map2 in range(N):
    #                B[map1][map2]+=(A[a][b][map1]*A[a][b][map2])
	A=K.batch_flatten(K.permute_dimensions(input, (2, 0, 1)))
	B=K.transpose(A)
	return K.dot(A,B)

def contentLossF(content,input):
	return K.sum(K.square(content-input))

#style already as gram matrix
def styleLossF(style,input):
	(x,y,N)=input.shape
	B=gramMat(input)
	C=K.sum(K.square(B-style))
	C/= (4.*(int(N)**2)*((int(x)*int(y))**2))
	return C
	
class contentLoss(losses.Loss):
	#pass cmodel.input as array. Pass model. Pass already predicted output.
	def __init__(self,content,input):
		super(contentLoss,self).__init__()
		self.name="Content Loss Function"
		self.content=content
		self.input=input
	def build_loss(self):
		return K.sum(K.square(self.content-self.input))
        
class styleLoss(losses.Loss):
	#predicted is already in gram matrix form
	def __init__(self,style,input):
		super(styleLoss,self).__init__()
		self.name="Style Loss Function"
		self.style=style
		self.input=input
	def build_loss(self):
		(x,y,N)=self.input.shape
		B=gramMat(self.input)
		C=K.sum(K.square(B-self.style))
		C/= (4.*(int(N)**2)*((int(x)*int(y))**2))
		return C

#change to pass in input image. Then run predict on truncated model.
    
def whiteNoise(rows,cols):
    noise_matrix = numpy.empty(shape=[rows,cols,3], dtype= numpy.uint8)
    for r in range(0,rows):
        for c in range(0,cols):
            noise_matrix[r][c][0] = randint(200,255)
            noise_matrix[r][c][1] = randint(200,255)
            noise_matrix[r][c][2] = randint(200,255)
    return noise_matrix

img = image.load_img('uva rotunda.jpg')
contentImage=image.img_to_array(img)

noise=whiteNoise(contentImage.shape[0],contentImage.shape[1])

img = image.load_img('pollockResized.jpg')
styleImage=image.img_to_array(img)
#styleImage=cv2.resize(styleImage,None,(contentImage.shape[0],contentImage.shape[1]))
#resize styleimage

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

print("loaded content and stylemodels 1-3")
contentBase = contentModel.predict(numpy.array([contentImage]))[0, :, :, :]
print("loaded content base")
styleBase1 = gramMat(styleModel1.predict(numpy.array([styleImage]))[0])
print("loaded stylebase1")
styleBase2 = gramMat(styleModel2.predict(numpy.array([styleImage]))[0])
print("loaded stylebase2")
styleBase3 = gramMat(styleModel3.predict(numpy.array([styleImage]))[0])
print("loaded stylebase3")

styleModel1=[]
styleModel2=[]
styleModel3=[]
contentModel=[]

print("deleted old models. Creating styleModel 4")
styleModel4 = VGG19(weights='imagenet',include_top= False,input_tensor=input_tensor)
for i in range(9):
    styleModel4.layers.pop()
styleModel4.outputs = [styleModel4.layers[-1].output]
styleModel4.layers[-1].outbound_nodes = []

styleBase4 = gramMat(styleModel4.predict(numpy.array([styleImage]))[0])

#block5_conv1, needt to repalce 8 with appropriate index offset
styleModel5 = VGG19(weights='imagenet',include_top= False,input_tensor=input_tensor)
for i in range(8):
    styleModel5.layers.pop()
styleModel5.outputs = [styleModel5.layers[-1].output]
styleModel5.layers[-1].outbound_nodes = []
styleBase5=gramMat(styleModel5.predict(numpy.array([styleImage]))[0]) 
styleModel5=[]   
#must create gram matrix of output of style image on style model
print("done loading models")


#print("laoded stylebase4")
#styleModel4=[]
#styleBase5 = gramMat(styleModel5.predict(numpy.array([styleImage])))

#sum style weights so that total approx = 1000
losses = [
    (contentLoss(contentBase,layer_dict["block4_conv2"][0, :, :, :]),2),
    (styleLoss(styleBase1,layer_dict["block1_conv1"][0, :, :, :]),250),
    (styleLoss(styleBase2,layer_dict["block2_conv1"][0, :, :, :]),250),
    (styleLoss(styleBase3,layer_dict["block3_conv1"][0, :, :, :]),250),
    (styleLoss(styleBase4,layer_dict["block4_conv1"][0, :, :, :]),250),
    (styleLoss(styleBase5,layer_dict["block5_conv1"][0, :, :, :]),300)
]

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
opt = Optimizer(trainedModel.input,losses)
print("Optimizer Initialized. Starting optimization")    
finaloutput=opt.minimize(seed_img=noise,max_iter=25,verbose=True,progress_gif_path='proggif.gif')[0]
print("finished")
pyplot.imshow(finaloutput)
pyplot.show() 
print("done with showing")
