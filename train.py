import datetime
import os
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import joint_transforms
from config import train_data
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
from PVTGate import PVTGate
from torch.backends import cudnn


cudnn.benchmark = True#寻找最适合当前配置的高效算法，优化运行效率
torch.manual_seed(2018)#设置固定生成随机数的种子，使得每次运行该 .py 文件时生成的随机数相同
torch.cuda.set_device(0)#将模型和数据加载到对应GPU上

##########################hyperparameters###############################
ckpt_path = './model'
exp_name = 'PVTGate'
args = {
    'iter_num': 1000,#训练迭代轮数
    'train_batch_size': 4,#训练的 mini-batch大小
    'last_iter': 0,#
    'lr': 1e-3,#初始学习率
    'lr_decay': 0.9,#学习率衰减
    'weight_decay': 0.0005,#权重衰减
    'momentum': 0.9,#优化方法动量
    'snapshot': ''
}
##########################data augmentation###############################数据增强
joint_transform = joint_transforms.Compose([     #多个图片变换列表#transforms.Compose就是将对图像处理的方法集中起来
    joint_transforms.RandomCrop(384, 384),  # 随机裁剪
    joint_transforms.RandomHorizontallyFlip(),#依概率（）水平翻转
    joint_transforms.RandomRotate(10)#随机旋转
])
img_transform = transforms.Compose([    #多个图像预处理列表
    transforms.ColorJitter(0.1, 0.1, 0.1),#改变图像的属性：修改亮度、对比度和饱和度
    transforms.ToTensor(),#将图片转换成Tensor
    #在做数据归一化之前必须要把PIL Image转成Tensor，而其他resize或crop操作则不需要。
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),#数据标准化，归一化，加快模型的收敛速度
])
target_transform = transforms.ToTensor()
##########################################################################
train_set = ImageFolder(train_data, joint_transform, img_transform, target_transform)# 使用ImageFolder读取数据
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=0, shuffle=True,drop_last=True)# 定义数据加载器



criterion = nn.BCEWithLogitsLoss().cuda()#损失函数
criterion_BCE = nn.BCELoss().cuda()
criterion_mse = nn.MSELoss().cuda()
criterion_mae = nn.L1Loss().cuda()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')
print (log_path)


def main():
    #############################PVTGate pretrained###########################
    model = PVTGate()
    ##############################Optim setting###############################
    net = model.cuda().train()
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])
    if len(args['snapshot']) > 0:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']
    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)


#########################################################################

def train(net, optimizer):#训练
    curr_iter = args['last_iter']
    while True:
        loss_record = AvgMeter()
        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']
            # data\binarizing\Variable
            inputs, labels = data
            labels[labels > 0.5] = 1
            labels[labels != 1] = 0
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            optimizer.zero_grad()
            output_pvt = net(inputs)

            ##########loss#############
            loss = criterion(output_pvt, labels)
            loss.backward()
            optimizer.step()
            loss_record.update(loss.item(), batch_size)

            #############log###############
            curr_iter += 1

            log = '[iter %d], [loss %.5f],[lr %.13f] ' % \
                  (curr_iter, loss_record.avg, optimizer.param_groups[1]['lr'])

            print(log)
            open(log_path, 'a').write(log + '\n')

            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))
                return
            ####################end###############


if __name__ == '__main__':
    main()
