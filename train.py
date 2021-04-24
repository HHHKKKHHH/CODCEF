import torch
import torch.nn.functional as F
from Src.utils.Dataloader import get_loader
from Src.FeaNet import SINet_ResNet50
import torch.optim as optim
from torch.autograd import Variable
from apex import amp
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# PPA Loss
def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def trainer_Innovation_2(train_loader, model, optimizer, epoch, opt,sw):
    """
    Training iteration
    :param train_loader:
    :param model:
    :param optimizer:
    :param epoch:
    :param opt:
    :param total_step:
    :return:
    """
    model.train()
    trainIt = enumerate(train_loader)
    for step, data_pack in trainIt:
        optimizer.zero_grad()
        images, gts = data_pack
       
        images = Variable(images).cuda(0)
        gts = Variable(gts).cuda(0)

        cam_sm, cam_im = model(images)
        loss_sm = structure_loss(cam_sm, gts)
        loss_im = structure_loss(cam_im, gts)
        loss_total = loss_sm + loss_im

        with amp.scale_loss(loss_total, optimizer) as scale_loss:
            scale_loss.backward()

        # clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if step % 10 == 0 :
            str = '[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}] => [Loss_s: {:.4f} Loss_i: {:0.4f}]'.format(datetime.now(), epoch, opt.epoch, step, loss_sm.data, loss_im.data)
            sw.writelines('{}|epoch:{:03d}/{:03d}|step:{:04d}|loss_sm:{:.4f}|loss_im:{:0.4f}'.format(datetime.now(), epoch, opt.epoch, step, loss_sm.data, loss_im.data) + '\n')
            print(str)

    save_path = opt.save_model
    os.makedirs(save_path, exist_ok=True)

    if (epoch+1) % opt.save_epoch == 0:
        torch.save(model.state_dict(), save_path + 'FeaNet_%d.pth' % (epoch+1))


parser = argparse.ArgumentParser()
parser.add_argument('--trainsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--save_epoch', type=int,
                    default=10)
parser.add_argument('--save_model', type=str,
                    default='./Snapshot/FeaNet/')
opt = parser.parse_args()

sw  = open('./out.txt','w')


trainDataRoot = 'Dataset/TrainDataset/Image/'
gtRoot = 'Dataset/TrainDataset/GT/'
model = SINet_ResNet50().cuda(0)
optimizer = optim.Adam(model.parameters(), lr=0.0001)


train_loader = get_loader(trainDataRoot, gtRoot, 23, opt.trainsize,shuffle=True, num_workers=12)
model, optimizer = amp.initialize(model, optimizer)
for epoch in range(opt.epoch):

    trainer_Innovation_2(train_loader, model, optimizer, epoch, opt,sw)
torch.save(model.state_dict(), opt.save_model + 'FeaNet_Final.pth')
sw.close()