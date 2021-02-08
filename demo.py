import torch
import torch.nn as nn
#from testRegDB import RegDBData
from data_loader import SYSUData
from model import *
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
from skimage.util import random_noise

import matplotlib.pyplot as plt

def concatImg(color, thermal):
    img = 255*np.ones((color.shape[0] + thermal.shape[0],
                    max(color.shape[1], thermal.shape[1]),
                    3), np.int)
    img[0:color.shape[0], -color.shape[1]:img.shape[1], :] = color
    img[color.shape[0]:img.shape[0], -thermal.shape[1]:img.shape[1], :] = thermal
    return img

def makeBorder(img, color=(0,255,0)):
    b = 5
    big = np.zeros((img.shape[0]+2*b,img.shape[1]+2*b, 3 ), np.int)
    big[:,:]=color
    big[b:-b, b:-b] = img
    return big

def display_multiple_img(images, rows = 1, cols=1, name='name.jpg'):
    figure, ax = plt.subplots(nrows=rows,ncols=cols )
    for ind,title in enumerate(images):
        ax.ravel()[ind].imshow(images[title])
        ax.ravel()[ind].set_title(title)
        ax.ravel()[ind].set_axis_off()
    plt.tight_layout()
    figure.savefig('results/'+name, dpi=figure.dpi)
    #plt.show()

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
unNormalize = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((288, 144)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddSaltPapper(object):
    def __init__(self, salt_vs_pepper=0.5, amount=.05):
        self.salt_vs_pepper = salt_vs_pepper
        self.amount = amount

    def __call__(self, tensor):
        return tensor
        #return torch.tensor(tensor.numpy())
        return np.array(random_noise(tensor, mode='salt')*255, tensor.dtype)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.salt_vs_pepper, self.amount)



transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((288, 144)),
    transforms.ToTensor(),
    normalize
])
transform_show = transforms.Compose([
    unNormalize,
    transforms.ToPILImage(),
])


transform_addNoise_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((288, 144)),
    transforms.ToTensor(),
    normalize,
    AddGaussianNoise(0., 0.5),

])


device = 'cuda' if torch.cuda.is_available() else 'cpu'
TEST_TYPE = 0
data_path = '../Datasets/SYSU-MM01/'
n_class = 296

fusion_function = 'add'
fusion_layer = '4'
net = embed_net(n_class, no_local= 'off', gm_pool =  'off', arch='resnet50',
                        fusion_layer=int(fusion_layer), fusion_function=fusion_function)
net.to(device)
checkpoint = torch.load("save_model/sysu_base_p4_n8_lr_0.1_seed_0_"+
                         fusion_function+fusion_layer +"_best.t"
                         #, map_location=torch.device('cpu')
                         )
# checkpoint = torch.load("save_model/sysu-Thermal/best.t"
#                         , map_location=torch.device('cpu')
#                         )

net.load_state_dict(checkpoint['net'])
net.eval()
gallset = SYSUData(data_path, dataFile='test', num_pos=1, transform=transform_train, isQG=True,
                           colorCam=[2, 5], irCam=[3])
queryset = SYSUData(data_path, dataFile='test', num_pos=1, transform=transform_test, isQG=True,
                            colorCam=[1, 4], irCam=[6])
def extractFeat():

    with torch.no_grad():
        #data = SYSUData ("", 1, transform_test)
        X = np.empty((0, 2048))
        y = np.empty(0)
        cR = np.empty(0)
        cI = np.empty(0)
        I = np.empty(0)
        pR = np.empty(0)
        pI = np.empty(0)
        m = {i: [] for i in gallset.getLabels()}
        for i, (x1, x2, y1, y2, _, _, _, _) in enumerate(gallset):
            m[y1] = np.append(m[y1], i)
        #Ls = np.random.choice(data.getLabels(), L)
        Ls = gallset.getLabels()
        for i in Ls:
            X2 = X1 = torch.Tensor()
            for j in m[i]:
                (x1, x2, y1, y2, c1, c2, rgb, ir) = gallset[j]
                X1 = torch.cat((X1, x1.unsqueeze(0)), dim=0)
                X2 = torch.cat((X2, x2.unsqueeze(0)), dim=0)
                y = np.append(y, [int(y1)], axis=0)
                cR = np.append(cR, [int(c1)], axis=0)
                cI = np.append(cI, [int(c2)], axis=0)
                I = np.append(I, [int(j)], axis=0)
                pR = np.append(pR, [rgb], axis=0)
                pI = np.append(pI, [ir], axis=0)

            X1 = Variable(X1.cuda())
            X2 = Variable(X2.cuda())
            i = 0
            while i < len(X1):
                j = min(i+20, len(X1))
                res = net(X1[i:j], X2[i:j], TEST_TYPE)
                X = np.append(X,res[0].cpu().numpy() , axis=0)
                i = j

    return X, y, cR, cI, I, pR, pI

def test():
    USE_NET = True
    gallery, y, cR, cI, I = None, None, None, None, None
    if USE_NET:
        gallery, y, cR, cI, I, pR, pI = extractFeat()
        np.save('feat/G.npy', gallery)
        np.save('feat/Y.npy', y)
        np.save('feat/C1.npy', cR)
        np.save('feat/C2.npy', cI)
        np.save('feat/I.npy', I)
        np.save('feat/pR.npy', pR)
        np.save('feat/pI.npy', pI)
    else:
        gallery= np.load('feat/G.npy')
        y = np.load('feat/Y.npy')
        cR = np.load('feat/C1.npy')
        cI = np.load('feat/C2.npy')
        I = np.load('feat/I.npy')
        pR = np.load('feat/pR.npy')
        pI = np.load('feat/pI.npy')


    with torch.no_grad():
        for j in range(0,len(queryset), 20):
            X2 = X1 = torch.Tensor()
            (x1, x2, y1, y2, c1, c2, rgb, ir) = queryset[j]
            qpid = int(rgb[-13:-9])

            X1 = torch.cat((X1, x1.unsqueeze(0)), dim=0)
            X2 = torch.cat((X2, x2.unsqueeze(0)), dim=0)

            X1cuda = Variable(X1.cuda())
            X2cuda = Variable(X2.cuda())
            resN = net(X1cuda, X2cuda, TEST_TYPE)
            res = resN[0].cpu().numpy()
            dis = np.matmul(gallery, np.transpose(res))
            index = np.squeeze(np.argsort(-1*dis, axis=0))
            total_images = 10
            images={}

            colorImgID = int(rgb[-8:-4])
            thermalImgID = int(ir[-8:-4])

            #cImg = np.array(Image.merge("RGB", (b, g, r)))
            cImg = np.array(transform_show(X1[0]))
            iImg = np.array(transform_show(X2[0]))
            #cImg = cImg.transpose(2, 0, 1)
            #cImg, iImg = np.array(Image.open(rgb)), np.array(Image.open(ir))
            if TEST_TYPE == 1:
                iImg[:, :, :] = 127
            if TEST_TYPE == 2:
                cImg[:, :, :] = 127
            images['q'+'\n id:'+str(qpid)] = concatImg(cImg, iImg)
            for i in range(total_images):
                color = np.array(Image.open(pR[index[i]]))
                thermal = np.array(Image.open(pI[index[i]]))
                if TEST_TYPE == 1:
                    thermal[:, :, :] = 127
                if TEST_TYPE == 2:
                    color[:, :, :] = 127
                gpid = int(pR[index[i]][-13:-9])
                images['r ' + str(i+1)+'\n id:'+str(gpid)] = \
                    makeBorder(concatImg(color, thermal), (255*(gpid!=qpid), 255*(gpid==qpid),0))


            display_multiple_img(images, 1, 1+total_images,
                                 name = str(qpid)+'('+str(c1)+','+str(c2)+')'+
                                 '(' + str(colorImgID) + ',' + str(thermalImgID) + ').jpg'
                                 )

test()
