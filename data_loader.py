import numpy as np
from PIL import Image
import torch.utils.data as data
from skimage.util import random_noise

class SYSUData(data.Dataset):
    def __init__(self, data_dir, num_pos=4, transform=None, colorIndex = None, thermalIndex = None, isQG = False,
                 dataFile='train', colorCam=[1,2,4,5], irCam=[3,6], modality=0):
        
        data_dir = '../Datasets/SYSU-MM01/'
        self.num_pos = num_pos
        # Load training images (path) and labels
        self.modality = modality
        #if self.modality == 0 or self.modality == 1:
        train_color_image = np.load(data_dir + dataFile+'_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + dataFile+'_rgb_resized_label.npy')
        self.train_color_camera = np.load(data_dir + dataFile+'_rgb_resized_camera.npy')
        self.files_rgb = np.load(data_dir + dataFile+'_rgb_path.npy')
        self.train_color_image = train_color_image
        self.cIndex = colorIndex
        self.colorCam = colorCam

        #if self.modality == 0 or self.modality == 2:
        train_thermal_image = np.load(data_dir + dataFile+'_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + dataFile+'_ir_resized_label.npy')
        self.train_thermal_camera = np.load(data_dir + dataFile+'_ir_resized_camera.npy')
        self.files_ir = np.load(data_dir + dataFile + '_ir_path.npy')
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.tIndex = thermalIndex
        self.irCam = irCam

        self.transform = transform
        self.shuffle()

    def addNoise(self, img):
        r = np.random.normal(0, .15, img.shape) * 255
        img = img + r
        img[img<0]=0
        img[img>255]=255
        img = img.astype(np.uint8)
        img = random_noise(img, mode='s&p', amount=0.1)
        img = img * 255

        r1, r2 = np.random.randint(int(img.shape[0]/3), size=2)
        c1, c2 = np.random.randint(int(img.shape[0]/3), size=2)
        img[min(r1,r2):max(r1, r2), min(c1, c2):max(c1, c2), :] = 0

        return img.astype(np.uint8)

    def __getitem__(self, index):
        labelId = int(index/self.num_pos)
        instanceId = int(index % self.num_pos)

        p = self.index[labelId][instanceId]
        img1,  target1 = self.train_color_image[p[0]],  self.train_color_label[p[0]]
        img2,  target2 = self.train_thermal_image[p[1]], self.train_thermal_label[p[1]]
        c1,         c2 = self.train_color_camera[p[0]], self.train_thermal_camera[p[1]]
        rgbPath, irPath = self.files_rgb[p[0]], self.files_ir[p[1]]

        #img1 = self.addNoise(img1)
        #img2 = self.addNoise(img2)

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2, c1 , c2, rgbPath, irPath

    def getImage(self, index):
        labelId = int(index / self.num_pos)
        instanceId = int(index % self.num_pos)

        p = self.index[labelId][instanceId]
        img1, target1 = self.train_color_image[p[0]], self.train_color_label[p[0]]
        img2, target2 = self.train_thermal_image[p[1]], self.train_thermal_label[p[1]]
        return img1, img2

    def shuffle(self):
        mC = {i: [] for i in np.unique(self.train_color_label)}
        mT = {i: [] for i in np.unique(self.train_thermal_label)}

        self.cIndex = np.array([], dtype=np.int)
        self.tIndex = np.array([], dtype=np.int)
        CC=0
        TT=0
        for (i, l) in enumerate(self.train_color_label):
            if self.train_color_camera[i] in self.colorCam:
                CC = CC + 1
                mC[l] = np.append(mC[l], i)
        for i in range(len(self.train_thermal_label)):
            if self.train_thermal_camera[i] in self.irCam:
                TT = TT + 1
                l = self.train_thermal_label[i]
                mT[l] = np.append(mT[l], i)

        if self.modality == 0 or self.modality == 1:
            for l, ll in mC.items():
                if len(ll) and l in mT and len(mT[l]):
                    self.cIndex = np.append(self.cIndex, ll, axis=0)
                    ll = np.random.choice(mT[l], len(ll))
                    self.tIndex = np.append(self.tIndex, ll, axis=0)

        elif self.modality == 2:
            for l, ll in mT.items():
                if len(ll) and l in mT and len(mC[l]):
                    self.tIndex = np.append(self.tIndex, ll, axis=0)
                    ll = np.random.choice(mC[l], len(ll))
                    self.cIndex = np.append(self.cIndex, ll, axis=0)

        self.cIndex = self.cIndex.astype(np.int)
        self.tIndex = self.tIndex.astype(np.int)
        self.index = np.column_stack((self.cIndex, self.tIndex))
        perfectSize = len(self.cIndex) - len(self.cIndex) % self.num_pos
        self.index=self.index[0:perfectSize, :]
        self.index = self.index.reshape(-1, self.num_pos, 2)
        #np.random.shuffle(self.index)
        return
    def __len__(self):
        return len(self.tIndex)

    def getLabels(self):
        return np.unique(self.train_color_label)
        
        
class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None, name='train', isQG = False):
        # Load training images (path) and labels
        data_dir = '../Datasets/RegDB/'
        train_color_list   = data_dir + 'idx/'+name+'_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/'+name+'_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)
        
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        if isQG:
            self.cIndex = np.arange(len(self.train_color_label))
            self.tIndex = np.arange(len(self.train_thermal_label))
            j= 0
            while j <len(self.train_color_label) - 1:
                k = j
                while (k + 1 < len(self.train_color_label)) \
                    and (self.train_thermal_label[k] == self.train_thermal_label[k+1]):
                    k = k + 1

                np.random.shuffle(self.tIndex[j : k + 1])
                j = k + 1

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
        
class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)
        
class TestDataOld(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(data_dir + test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)        
def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label
