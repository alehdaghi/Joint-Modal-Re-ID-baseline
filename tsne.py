import torch
import torch.nn as nn
#from testRegDB import RegDBData
from data_loader import SYSUData
from model import *
import torchvision.transforms as transforms
import numpy as np
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist
from sklearn.manifold._t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
from torch.autograd import Variable
import pandas as pd

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((288, 144)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((288, 144)),
    transforms.ToTensor(),
    normalize,
])

sns.set(rc={'figure.figsize':(11.7,8.27)})

MACHINE_EPSILON = np.finfo(np.double).eps
n_components = 2
perplexity = 30

def fit(X):
    n_samples = X.shape[0]

    # Compute euclidean distance
    distances = pairwise_distances(X, metric='euclidean', squared=True)

    # Compute joint probabilities p_ij from distances.
    P = _joint_probabilities(distances=distances, desired_perplexity=perplexity, verbose=False)

    # The embedding is initialized with iid samples from Gaussians with standard deviation 1e-4.
    X_embedded = 1e-4 * np.random.mtrand._rand.randn(n_samples, n_components).astype(np.float32)

    # degrees_of_freedom = n_components - 1 comes from
    # "Learning a Parametric Embedding by Preserving Local Structure"
    # Laurens van der Maaten, 2009.
    degrees_of_freedom = max(n_components - 1, 1)

    return _tsne(P, degrees_of_freedom, n_samples, X_embedded=X_embedded)

def _tsne(P, degrees_of_freedom, n_samples, X_embedded):
    params = X_embedded.ravel()
    obj_func = _kl_divergence
    params = _gradient_descent(obj_func, params, [P, degrees_of_freedom, n_samples, n_components])
    X_embedded = params.reshape(n_samples, n_components)
    return X_embedded

def _kl_divergence(params, P, degrees_of_freedom, n_samples, n_components):
    X_embedded = params.reshape(n_samples, n_components)

    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # Kullback-Leibler divergence of P and Q
    kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))

    # Gradient: dC/dY
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order='K'),
                         X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c


    return kl_divergence, grad

def _gradient_descent(obj_func, p0, args, it=0, n_iter=1000,
                      n_iter_check=1, n_iter_without_progress=300,
                      momentum=0.8, learning_rate=200.0, min_gain=0.01,
                      min_grad_norm=1e-7):
    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = i = it

    for i in range(it, n_iter):
        error, grad = obj_func(p, *args)
        grad_norm = linalg.norm(grad)
        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update
        print("[t-SNE] Iteration %d: error = %.7f,"
              " gradient norm = %.7f"
              % (i + 1, error, grad_norm))

        if error < best_error:
            best_error = error
            best_iter = i
        elif i - best_iter > n_iter_without_progress:
            break

        if grad_norm <= min_grad_norm:
            break


    return p



L = 10
fusion_function = 'cat'
fusion_layer = '4'
def extractFeat():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        #data = SYSUData ("", 1, transform_test)
        TEST_TYPE = 0
        data_path = '../Datasets/SYSU-MM01/'
        data = SYSUData(data_path, dataFile='test', num_pos=1, transform=transform_test,
                            isQG=True, modality=TEST_TYPE)


        n_class = 296
        net = embed_net(n_class, no_local='off', gm_pool='off', arch='resnet50',
                        fusion_layer=int(fusion_layer), fusion_function=fusion_function)
        print(net.count_params())
        exit(0)
        checkpoint = torch.load("save_model/sysu_base_p4_n8_lr_0.1_seed_0_"+
                                fusion_function+fusion_layer +"_best.t"
                                #, map_location=torch.device('cpu')
                                )
        # checkpoint = torch.load("save_model/sysu_base_p4_n8_lr_0.1_seed_0_cat0_Thermal_epoch_80.t"
        #                         # , map_location=torch.device('cpu')
        #                         )


        net.to(device)
        net.load_state_dict(checkpoint['net'])

        net.eval()

        X = np.empty((0, 2048))
        y = np.empty(0)
        cR = np.empty(0)
        cI = np.empty(0)
        m = {i: [] for i in data.getLabels()}
        for i, (x1, x2, y1, y2, _, _, _, _) in enumerate(data):
            m[y1] = np.append(m[y1], i)
        #Ls = np.random.choice(data.getLabels(), L)
        Ls = range(L)
        for i in Ls:
            X2 = X1 = torch.Tensor()
            for j in m[i]:
                (x1, x2, y1, y2, c1, c2, _1, _2) = data[j]
                X1 = torch.cat((X1, x1.unsqueeze(0)), dim=0)
                X2 = torch.cat((X2, x2.unsqueeze(0)), dim=0)
                y = np.append(y, [int(y1)], axis=0)
                cR = np.append(cR, [int(c1)], axis=0)
                cI = np.append(cI, [int(c2)], axis=0)

            X1 = Variable(X1.cuda())
            X2 = Variable(X2.cuda())

            #x1 = x1.unsqueeze(0)
            #x2 = x2.unsqueeze(0)
            i = 0
            while i < len(X1):
                j = min(i+20, len(X1))
                res = net(X1[i:j], X2[i:j], TEST_TYPE)
                X = np.append(X,res[1].cpu().numpy() , axis=0)
                i = j

    return X, y, cR, cI




USE_NET = True
if USE_NET:
    X, y, cR, cI = extractFeat()
    X_embedded = fit(X)
    np.save('X.npy', X_embedded)
    np.save('y.npy', y)
    np.save('cR.npy', cR)
    np.save('cI.npy', cI)
else:
    X_embedded = np.load('X.npy')
    y = np.load('y.npy')
    cR = np.load('cR.npy')
    cI = np.load('cI.npy')

palette = sns.color_palette("muted", L)
df = pd.DataFrame(columns=['x','y','id','cR', 'cI'])
df['x'] = X_embedded[:,0]
df['y'] = X_embedded[:,1]
df['id'] = y+1
df['cR'] = cR
df['cI'] = cI
sns.set_theme()
p = sns.hls_palette(L, h=.5)
plt.title("Visualize features of "+str(L) +" identity fusion:"+fusion_function+" layer:"+fusion_layer )
g=sns.scatterplot(data=df, x='x', y='y', style= 'id',size='cI' , hue='cR', palette='hls')
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.show()
