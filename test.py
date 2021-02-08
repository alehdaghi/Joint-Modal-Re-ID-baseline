from __future__ import print_function
import argparse
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
from utils import *
import pdb

TEST_TYPE = 0 # 0 Fusion, 1: Color, 2:Thermal

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline: resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=16, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='base', type=str,
                    metavar='m', help='method type: base or awg')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor for sysu')
parser.add_argument('--tvsearch', action='store_true', help='whether thermal to visible search on RegDB')
parser.add_argument('--fusion_function', default='cat', type=str, help='cat or add')
parser.add_argument('--fusion_layer', default=0, type=int,
                    help='the layer fro which the fusion function applied')


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dataset = args.dataset
if dataset == 'sysu':
    data_path = '../Datasets/SYSU-MM01/ori_data/'
    n_class = 296
    test_mode = [1, 2]
elif dataset =='regdb':
    data_path = '../Datasets/RegDB/'
    n_class = 206
    test_mode = [2, 1]
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0 
pool_dim = 2048

print('==> Building model..')
if args.method =='base':
    net = embed_net(n_class, no_local= 'off', gm_pool =  'off', arch=args.arch,
                    fusion_layer=int(args.fusion_layer), fusion_function=args.fusion_function)
else:
    net = embed_net(n_class, no_local= 'on', gm_pool = 'on', arch=args.arch,
                    fusion_layer=int(args.fusion_layer), fusion_function=args.fusion_function)
net.to(device)    
cudnn.benchmark = True

checkpoint_path = args.model_path

if args.method =='id':
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((args.img_h,args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h,args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()



def extract_gall_feat(gall_loader):
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.empty((0, 2048))
    gall_feat_fc = np.empty((0, 2048))
    gall_label = np.empty(0)
    gall_cam = np.empty(0)
    pid = np.empty(0)

    with torch.no_grad():
        for batch_idx, (input1, input2, label1, label2, _, _, _1, _2) in enumerate(gall_loader):
            batch_num = input1.size(0)

            if torch.cuda.is_available():
                if TEST_TYPE == 0 or TEST_TYPE == 1:
                    input1 = Variable(input1.cuda())
                if TEST_TYPE == 0 or TEST_TYPE == 2:
                    input2 = Variable(input2.cuda())

            feat_pool, feat_fc = net(input1, input2, TEST_TYPE)
            gall_feat_pool = np.append(gall_feat_pool, feat_pool.detach().cpu().numpy(), axis=0)
            gall_feat_fc = np.append(gall_feat_fc, feat_fc.detach().cpu().numpy(), axis=0)
            gall_label = np.append(gall_label, label1, axis=0)
            gall_cam = np.append(gall_cam, -1 * _, axis=0)
            temp1 = [int(i[-13:-9]) for i in _1]
            temp2 = [int(i[-13:-9]) for i in _2]
            if temp1 != temp2:
                print('FUCK!')
            pid = np.append(pid, temp1, axis=0)
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat_pool, gall_feat_fc, gall_label, gall_cam, pid
    
def extract_query_feat(query_loader):
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0

    query_feat_pool = np.empty((0, 2048))
    query_feat_fc = np.empty((0, 2048))
    query_label = np.empty(0)
    query_cam = np.empty(0)
    pid = np.empty(0)
    with torch.no_grad():
        for batch_idx, (input1, input2, label1, label2, _, _, _1, _2) in enumerate(query_loader):
            batch_num = input1.size(0)
            if torch.cuda.is_available():
                if TEST_TYPE == 0 or TEST_TYPE == 1:
                    input1 = Variable(input1.cuda())
                if TEST_TYPE == 0 or TEST_TYPE == 2:
                    input2 = Variable(input2.cuda())

            feat_pool, feat_fc = net(input1, input2, TEST_TYPE)
            query_feat_pool = np.append(query_feat_pool, feat_pool.detach().cpu().numpy(), axis=0)
            query_feat_fc = np.append(query_feat_fc, feat_fc.detach().cpu().numpy(), axis=0)
            query_label = np.append(query_label, label1, axis=0)
            query_cam = np.append(query_cam, -1 * _, axis=0)
            temp1 = [int(i[-13:-9]) for i in _1]
            temp2 = [int(i[-13:-9]) for i in _2]
            if temp1 != temp2:
                print('FUCK!')
            pid = np.append(pid, temp1, axis=0)


            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat_pool, query_feat_fc, query_label, query_cam, pid


if dataset == 'sysu':

    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = checkpoint_path + args.resume
        # model_path = checkpoint_path + 'sysu_awg_p4_n8_lr_0.1_seed_0_best.t'
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # testing set
    queryset = SYSUData(data_path, dataFile='test', num_pos=1, transform=transform_train, isQG=True, colorCam=[1, 4],
                        irCam=[6])
    gallset = SYSUData(data_path, dataFile='test', num_pos=1, transform=transform_train, isQG=True, colorCam=[2, 5],
                       irCam=[3])

    nquery = len(queryset.getLabels())
    ngall = len(gallset.getLabels())
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(queryset.getLabels())), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gallset.getLabels())), ngall))
    print("  ------------------------------")

    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, drop_last=True, num_workers=args.workers)
    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, drop_last=False, num_workers=args.workers )
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))


    query_feat_pool, query_feat_fc, query_label, query_cam, qid  = extract_query_feat(query_loader)
    gall_feat_pool, gall_feat_fc, gall_label, gall_cam, gid = extract_gall_feat(gall_loader)
    np.save('feat/query_feat_pool.npy', query_feat_pool)
    np.save('feat/query_feat_fc.npy', query_feat_fc)
    np.save('feat/gall_feat_pool.npy', gall_feat_pool)
    np.save('feat/gall_feat_fc.npy', gall_feat_fc)
    np.save('feat/qid.npy', qid)
    np.save('feat/gid.npy', gid)
    distmat = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
    distmat_att = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

    cmc, mAP, mINP = eval_sysu(-distmat, qid, gid, query_cam, gall_cam)
    cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, qid, gid, query_cam, gall_cam)

    print(
        'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))


elif dataset == 'regdb':

    for trial in range(10):
        test_trial = trial +1
        #model_path = checkpoint_path +  args.resume
        model_path = checkpoint_path + 'regdb_awg_p4_n8_lr_0.1_seed_0_trial_{}_best.t'.format(test_trial)
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])

        # training set
        trainset = RegDBData(data_path, test_trial, transform=transform_train)
        # generate the idx of each person identity
        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

        # testing set
        query_img, query_label = process_test_regdb(data_path, trial=test_trial, modal='visible')
        gall_img, gall_label = process_test_regdb(data_path, trial=test_trial, modal='thermal')

        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        nquery = len(query_label)
        ngall = len(gall_label)

        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        print('Data Loading Time:\t {:.3f}'.format(time.time() - end))


        query_feat_pool, query_feat_fc = extract_query_feat(query_loader)
        gall_feat_pool,  gall_feat_fc = extract_gall_feat(gall_loader)

        np.save('feat/query_feat_pool.npy', query_feat_pool)
        np.save('feat/query_feat_fc.npy', query_feat_fc)
        np.save('feat/gall_feat_pool.npy', gall_feat_pool)
        np.save('feat/gall_feat_fc.npy', gall_feat_fc)

        if args.tvsearch:
            # pool5 feature
            distmat_pool = np.matmul(gall_feat_pool, np.transpose(query_feat_pool))
            cmc_pool, mAP_pool, mINP_pool = eval_regdb(-distmat_pool, gall_label, query_label)

            # fc feature
            distmat = np.matmul(gall_feat_fc , np.transpose(query_feat_fc))
            cmc, mAP, mINP = eval_regdb(-distmat,gall_label,  query_label )
        else:
            # pool5 feature
            distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
            cmc_pool, mAP_pool, mINP_pool = eval_regdb(-distmat_pool, query_label, gall_label)

            # fc feature
            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)


        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
            all_mINP_pool = mINP_pool
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
            all_mINP_pool = all_mINP_pool + mINP_pool

        print('Test Trial: {}'.format(trial))
        print(
            'FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))

# cmc = all_cmc / 10
# mAP = all_mAP / 10
# mINP = all_mINP / 10
#
# cmc_pool = all_cmc_pool / 10
# mAP_pool = all_mAP_pool / 10
# mINP_pool = all_mINP_pool / 10
# print('All Average:')
# print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
#         cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
# print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
#     cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))

