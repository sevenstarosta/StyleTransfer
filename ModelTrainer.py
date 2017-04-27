from keras.applications import VGG19
from keras.models import Sequential
from keras.layers import Input
from keras.preprocessing import image
from random import randint
from keras.utils import plot_model
from vis.optimizer import Optimizer
from vis import losses
from matplotlib import pyplot
import numpy
#import cv2

class contentLoss(losses.Loss):
    #pass cmodel.input as array. Pass model. Pass already predicted output.
    def __init__(self,layer,predicted):
        super(contentLoss,self).__init__()
        self.name="Content Loss Function"
        self.layer=layer
        self.predicted=predicted
        
    def build_loss(self):
        A=self.layer.get_output_at(0)
        B=self.predicted
        (l,x,y,N)=A.shape
        C=0
        for a in range(x):
            for b in range(y):
                for map in range(N):
                    C+=(A[0][a][b][map]-B[0][a][b][map])**2
        C/=2
        return C
        
class styleLoss(losses.Loss):
    #predicted is already in gram matrix form
    def __init__(self,layer,predicted):
        super(styleLoss,self).__init__()
        self.name="Style Loss Function"
        self.layer=layer
        self.predicted=predicted
        
    def build_loss(self):
        #need to build 4d array from layer.
        pred=self.layer.get_output_at(0)
        (l,x,y,N)=pred.shape
        A=gramMat(pred)
        C=0
        for x in range(N):
            for y in range(N):
                C+=(A[x][y]-self.predicted[x][y])**2
        C/= (4*(N**2)*((x*y)**2))
        return C
        
def gramMat(A):
    (l,x,y,N)=A.shape
    B=[[0 for a in range(N)]for b in range(N)]
    for map1 in range(N):
        for map2 in range(N):
            for a in range(x):
                for b in range(y):
                    B[map1][map2]+=(A[0][a][b][map1]*A[0][a][b][map2])
    return B

#change to pass in input image. Then run predict on truncated model.
    
def whiteNoise(rows,cols):
    noise_matrix = numpy.empty(shape=[rows,cols,3], dtype= numpy.uint8)
    for r in range(0,rows):
        for c in range(0,cols):
            noise_matrix[r][c][0] = randint(200,255)
            noise_matrix[r][c][1] = randint(200,255)
            noise_matrix[r][c][2] = randint(200,255)
    return noise_matrix

img = image.load_img('eiffel128.png')
contentImage=image.img_to_array(img)

noise=whiteNoise(contentImage.shape[0],contentImage.shape[1])

img = image.load_img('starrynight128.jpg')
styleImage=image.img_to_array(img)
#styleImage=cv2.resize(styleImage,None,(contentImage.shape[0],contentImage.shape[1]))
#resize styleimage

input_tensor = Input(shape=contentImage.shape)

trainedModel = VGG19(weights='imagenet',include_top= False,input_tensor=input_tensor)

layer_dict = dict([(layer.name, layer) for layer in trainedModel.layers[1:]])

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
#styleModel5 = VGG19(weights='imagenet',include_top= False,input_tensor=input_tensor)
#for i in range(8):
#    styleModel5.layers.pop()
#styleModel5.outputs = [styleModel5.layers[-1].output]
#styleModel5.layers[-1].outbound_nodes = []
    
#must create gram matrix of output of style image on style model
contentBase = contentModel.predict(numpy.array([contentImage]))
styleBase1 = gramMat(styleModel1.predict(numpy.array([styleImage])))
styleBase2 = gramMat(styleModel2.predict(numpy.array([styleImage])))
styleBase3 = gramMat(styleModel3.predict(numpy.array([styleImage])))
styleBase4 = gramMat(styleModel4.predict(numpy.array([styleImage])))
#styleBase5 = gramMat(styleModel5.predict(numpy.array([styleImage])))

#sum style weights so that total approx = 1000
losses = [
    (contentLoss(layer_dict["block4_conv2"],contentBase),1),
    (styleLoss(layer_dict["block1_conv1"],styleBase1),250),
    (styleLoss(layer_dict["block2_conv1"],styleBase2),250),
    (styleLoss(layer_dict["block3_conv1"],styleBase3),250),
    (styleLoss(layer_dict["block4_conv1"],styleBase4),250)
]

opt = Optimizer(trainedModel.input,losses)
print("starting optimization")    
finaloutput=opt.minimize(seed_img=noise,max_iter=40,verbose=False,progress_gif_path='proggif.gif')[0]
print("finished")
pyplot.imshow(finaloutput)
pyplot.show()
print("done with showing")
