import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc  # NOTES: pip install scipy == 1.2.2
from Src.FeaNet import SINet_ResNet50
from Src.utils.Dataloader import test_dataset
from Src.utils.trainer import eval_mae, numpy2tensor


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str,
                    default='./Snapshot/FeaNet/SINet_Final.pth')
parser.add_argument('--test_save', type=str,
                    default='./Result/FeaNet/')
opt = parser.parse_args()

model = SINet_ResNet50().cuda()
model.load_state_dict(torch.load(opt.model_path))
model.eval()

for dataset in ['COD10K','CAMO','CHAMELEON']:
    save_path = opt.test_save + dataset + '/'
    os.makedirs(save_path, exist_ok=True)

    test_loader = test_dataset('./Dataset/TestDataset/{}/Image/'.format(dataset),
                               './Dataset/TestDataset/{}/GT/'.format(dataset), opt.testsize)
    img_count = 0
    totalMae = 0
    for iteration in range(test_loader.size):
        image, gt, name = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        cam, _ = model(image)

        cam = F.upsample(cam, size=gt.shape, mode='bilinear', align_corners=True)
        cam = cam.sigmoid().data.cpu().numpy().squeeze()
        # normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        misc.imsave(save_path+"sm_"+name, cam)
        
        mae = eval_mae(numpy2tensor(cam), numpy2tensor(gt))
        img_count += 1
        totalMae+=mae
        # coarse score
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE: {}, AV_MAE: {}'.format(dataset, name, img_count,
                                                                           test_loader.size, mae,totalMae/img_count))

       

print("\n[Congratulations! Testing Done]")
