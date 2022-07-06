import numpy as np
import copy
import torch
from torchvision import datasets, transforms
import numpy as np
from torch import nn, optim
from matplotlib import pyplot as plt
import time
import math
import os
import skimage.io 
import skimage.segmentation
import copy
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers




if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', device)


## 교수님 코드

# 수정해야할 함수
class mnist_data():
    def __init__(self, **kwargs):
        transform = transforms.Compose([
            transforms.ToTensor()])

        self.kwargs = kwargs

        self.train_loaders = torch.utils.data.DataLoader(
            datasets.MNIST(root='./data', train=True, transform=transform, download=True), **kwargs)

        self.test_loaders = torch.utils.data.DataLoader(
            datasets.MNIST(root='./data', train=False, transform=transform, download=True), **kwargs)

    def train_loader(self):
        return self.train_loaders

    def test_loader(self):
        return self.test_loaders

    def save(self, loader_type):

        if loader_type == 'training':
            loader = self.train_loaders
            temp = np.zeros((60000,28,28))
            label = np.zeros(60000)
            center = np.zeros((60000,2))
        else:
            loader = self.test_loaders
            temp = np.zeros((10000,28,28))
            label = np.zeros(10000)
            center = np.zeros((10000,2))
        with torch.no_grad():
            if loader_type == 'training':
                ob=30000
            elif loader_type == 'test':
                ob=5000
            while True:
                i = 0
                for data, target in loader:
                    

                    if label[i]<1:
                        if label.sum()<ob:
                            data= data.view(28,28)
                            p = np.random.randint(2)
                            if p==1:
                                center[i] ,mark_data = mark_stamp(data.clone().numpy()*255,'stamp1')             
                                #plt.imshow(mark_data)
                                #print(target)
                                #np.save(f'C:\song\stamp\mark_{i:.0f}',mark_data)
                                label[i] = 1 
                            if p != 1:
                            
                                mark_data = data.clone().numpy()*255                                
                        if label.sum()>=ob :
                            data= data.view(28,28)
                            mark_data = data
                            #np.save(f'C:\song\stamp\mark_{i:.0f}',data)


                        temp[i] = mark_data
                    i = i+1
                    
                if label.sum()>=ob:break                    
                
                
            print(label.sum())    
            if loader_type == 'training':
                                
                np.save('train_stamp.npy',temp)
                np.save('train_label.npy',label)
                np.save('train_center.npy',center)
            
            else:
                                
                np.save('test_stamp.npy',temp)
                np.save('test_label.npy',label)
                np.save('test_center.npy',center)
  
                






# 도장 찍는 함수
def mark_stamp(img, type = 'stamp1'):
    shapes = img.shape
    center = np.random.randint(1,(shapes[0]-2),(2))

    if type =='stamp5':
        mark = np.zeros([3, 3])
        mark[1, :] = 255
        mark[:, 1] = 255

        img[(center[0] - 1):(center[0] + 2), (center[1] - 1):(center[1] + 2)] = mark

    elif type == 'stamp1':
        mark = np.zeros([2, 3])
        mark[0, 0] = 255
        mark[1, 1] = 255
        mark[0, 2] = 255

        img[(center[0]):(center[0] + 2), (center[1] - 1):(center[1] + 2)] = mark

    elif type == 'stamp2':
        mark = np.zeros([2, 4])
        mark[0, 1] = 255
        mark[1, 1] = 255
        mark[0, 2] = 255
        mark[1, 2] = 255

        img[(center[0]):(center[0] + 2), (center[1] - 1):(center[1] + 3)] = mark

    elif type == 'stamp3':
        mark = np.zeros([2, 4])
        mark[0, 1] = 255
        mark[1, 0] = 255
        mark[1, 2] = 255
        mark[0, 3] = 255

        img[(center[0]):(center[0] + 2), (center[1] - 1):(center[1] + 3)] = mark

    elif type == 'stamp4':
        center = np.random.randint(1, (shapes[0] - 14), (2))
        img[center[0], center[1]] = 255
        img[center[0] + 14, center[1]] = 255
        img[center[0] + 14, center[1] + 14] = 255
        img[center[0], center[1] + 14] = 255

    return center, img

# 측정용
def seperate_regions(O_img, trick ,center, type):
    temp = 0
    img = copy.deepcopy(O_img)
    if np.all(center !=(0,0)):
        if type == 'stamp1':

            mark = np.zeros([2, 3])
            #mark.fill(-1)
            mark.fill(1)
            mark[0, 0] = 1
            mark[1, 1] = 1
            mark[0, 2] = 1

            img_1 = copy.deepcopy(img[(center[0]):(center[0] + 2), (center[1] - 1):(center[1] + 2)]*mark)
            img_1
            img_1 = img_1.reshape(-1)

            img[(center[0]):(center[0] + 2), (center[1] - 1):(center[1] + 3)] = 999999

            mark = np.zeros([2, 3])
            mark[0, 0] = 255
            mark[1, 1] = 255
            mark[0, 2] = 255
            # 추가
            
            mark = np.zeros([2, 3])
            mark.fill(-1)
            # mark.fill(1)
            mark[0, 0] = 1
            mark[1, 1] = 1
            mark[0, 2] = 1

            img_2 = copy.deepcopy(trick[(center[0]):(center[0] + 2), (center[1] - 1):(center[1] + 2)]*mark)
            img_2
            img_2 = img_2.reshape(-1)


            mark = np.zeros([2, 3])
            mark[0, 0] = 255
            mark[1, 1] = 255
            mark[0, 2] = 255
            if np.all(img_2 == mark.reshape(-1)):
                temp = 1


        elif type == 'stamp2':
            mark = np.zeros([2, 4])
            mark.fill(-1)
            mark[0, 1] = 1
            mark[1, 1] = 1
            mark[0, 2] = 1
            mark[1, 2] = 1

            img_1 = copy.deepcopy(img[(center[0]):(center[0] + 2), (center[1] - 1):(center[1] + 3)] * mark)
            img_1 = img_1.reshape(-1)

            img[(center[0]):(center[0] + 2), (center[1] - 1):(center[1] + 3)] = 999999

        elif type == 'stamp3':
            mark = np.zeros([2, 4])
            mark.fill(-1)
            mark[0, 1] = 1
            mark[1, 0] = 1
            mark[1, 2] = 1
            mark[0, 3] = 1

            img_1 = copy.deepcopy(img[(center[0]):(center[0] + 2), (center[1] - 1):(center[1] + 3)] * mark)
            img_1 = img_1.reshape(-1)

            img[(center[0]):(center[0] + 2), (center[1] - 1):(center[1] + 3)] = 999999

        elif type == 'stamp5':
            img_1 = copy.deepcopy(img[(center[0] - 1):(center[0] + 2), (center[1] - 1):(center[1] + 2)]) * np.array(
                [[-1, 1, -1], [1, 1, 1], [-1, 1, -1]])
            # img_1 = copy.deepcopy(img[(center[0] - 1):(center[0] + 2), (center[1] - 1):(center[1] + 2)]) * np.array(
            #     [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            img_1 = img_1.reshape(-1)

            img[(center[0] - 1):(center[0] + 2), (center[1] - 1):(center[1] + 2)] = 999999

        elif type == 'stamp4':
            img_1 = np.array(copy.deepcopy(
                [img[center[0], center[1]], img[center[0] + 14, center[1]], img[center[0], center[1] + 14],
                 img[center[0] + 14, center[1] + 14]]))
            img_1 = img_1.reshape(-1)

            img[center[0], center[1]] = 999999
            img[center[0] + 14, center[1]] = 999999
            img[center[0] + 14, center[1] + 14] = 999999
            img[center[0], center[1] + 14] = 999999

        img = img.reshape(-1)
        img = img[img != 999999]


        img_av = np.average(img)
        img_std = np.std(img)

        # print('aver: ', img_av , 'std: ', img_std)
        #
        # plt.hist(img.reshape(-1))
        # plt.show()



        #normalization
        if img_std < 10**(-4):
            img_std = 10**(-3)
            img = (O_img - img_av) / img_std
            img_1 = (img_1 - img_av)/img_std

        else:
            img = (O_img - img_av) / img_std
            img_1 = (img_1 - img_av)/img_std
        # print('aver: ',np.average(img), 'std: ', np.std(img))
        # plt.hist(img.reshape(-1))
        # plt.show()

        img = np.clip(img,-100,100)
        img_1 = np.clip(img_1,-100,100)
    else:
        img_1 = np.zeros([28,28])
        img = O_img
        temp = 0

    return np.average(img_1), img, temp

def to_rgb(x):
    x_rgb = np.zeros((x.shape[0], 28, 28, 3))
    for i in range(3):
        x_rgb[..., i] = x[..., 0]
    return x_rgb

## code


# normalize
x_train = np.load('c:/song/train_stamp.npy',allow_pickle=True)/255
train_center = np.load('c:/song/train_center.npy',allow_pickle=True)
y_train = np.load('c:/song/train_label.npy').reshape(-1,1)
x_test = np.load('c:/song/test_stamp.npy')/255
test_center = np.load('c:/song/test_center.npy',allow_pickle=True)
y_test = np.load('c:/song/test_label.npy').reshape(-1,1)
fig, m_axs = plt.subplots(2,5, figsize = (12,6))

for i in range(1,11):
      plt.subplot(2,5,i)
      plt.imshow(x_train[i])
      plt.xlabel(y_train[i].item())

plt.show()



x_train = to_rgb(x_train.reshape(-1,28,28,1))
x_test = to_rgb(x_test.reshape(-1,28,28,1))

model = keras.Sequential(
    [
     keras.Input(shape=(28,28,3)),
     layers.Conv2D(16, 3, activation='relu'),
     layers.MaxPooling2D(),
     layers.Flatten(),
     layers.Dense(2)
    ]
)

model.compile(
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  optimizer=keras.optimizers.Adam(),
  metrics=['accuracy']
)


import wandb
from wandb.keras import WandbCallback

wandb.init(project='mnist_example', entity='sharpit')


model.fit(
        x_train, 
        y_train, 
        epochs=10, 
        batch_size=32, 
        validation_data = (x_test, y_test),
        callbacks = [WandbCallback(log_weights=(True), log_gradients=(True), training_data=(x_train,y_train))])


PATH = 'C:\song\stamp_weight/'
model.save(PATH+'CNN_model.h5')
model.save_weights(PATH+'CNN_model_weight.h5')


import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import random


from lime.wrappers.scikit_image import SegmentationAlgorithm

predictions = model.predict(x_test)
segmenter = SegmentationAlgorithm('slic',
                                  n_segments=50,
                                  compactness=1,
                                  sigma=1)

start = time.time()
fig, m_axs = plt.subplots(20,5, figsize = (20,80))

explainer = lime_image.LimeImageExplainer(random_state=42)

ac = []
score = []
im = []
im_o = []
mask_o = []
heat = []
heat_o = []
i=0
p=0
while 1:
    if np.argmax(predictions[i]) == 1 :
        
        
        explanation = explainer.explain_instance(
             x_test[i], 
             model.predict,
             top_labels=1,
             num_samples=1000,
             segmentation_fn=segmenter)

        image, mask = explanation.get_image_and_mask(
                 model.predict(
                      x_test[i].reshape((1,28,28,3))
                 ).argmax(axis=1)[0],
                 positive_only=True, 
                 num_features = 1,
                 hide_rest=True)
        
        plt.subplot(20,5,p+1)

        #Select the same class explained on the figures above.
        ind =  explanation.top_labels[0]

        #Map each explanation weight to the corresponding superpixel
        dict_heatmap = dict(explanation.local_exp[ind])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 

        #Plot. The visualization makes more sense if a symmetrical colorbar is used.
        plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
        plt.colorbar()
        plt.xlabel(explanation.score)
        #plt.title(np.argmax(predictions[i]))
        heatmap_o = heatmap
        heatmap[np.isnan(heatmap)] = 0
        p = p + 1
        a,b,ev = seperate_regions(heatmap,(x_test[i][:,:,0]*mask)*255, test_center[i].astype('int'), 'stamp1')
        ac.append(ev)
        if ev ==1:
            score.append(a)
            im.append(b)
            im_o.append((x_test[i][:,:,0]*mask)*255)
            mask_o.append(mask)
            heat.append(heatmap)
            heat_o.append(heatmap_o)
            
    i= i+1
    if p == 100:break
end = time.time()
print(f"{end - start:.5f} sec")

print(score[0:10])