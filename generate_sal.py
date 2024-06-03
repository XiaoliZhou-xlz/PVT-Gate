import numpy as np
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from config import test_data
from misc import check_mkdir, crf_refine
from PVTGate import PVTGate
torch.manual_seed(2018)
torch.cuda.set_device(0)
import time

ckpt_path = './model'
exp_name = 'model_gatenet'

args = {
    'snapshot': '',
    'crf_refine': False,
    'save_results': True
}

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

to_test = {'test':test_data}

def main():
    #########################Load##########################
    net = PVTGate().cuda()
    print ('load snapshot \'%s\' for testing' % args['snapshot'])
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot']+ '100000.pth'),map_location={'cuda:1': 'cuda:1'}))
    net.eval()

    ########################################################
    with torch.no_grad():

        for name, root in to_test.items():
            check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot'])))
            root1 = os.path.join(root,'Mask')
            img_list = [os.path.splitext(f)[0] for f in os.listdir(root1) if f.endswith('.png')]
            print(img_list)
            for idx, img_name in enumerate(img_list):
                print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                img1 = Image.open(os.path.join(root,'Image/'+img_name +'.jpg')).convert('RGB')
                img = img1
                w,h = img1.size
                img1 = img1.resize([384, 384])
                img_var = Variable(img_transform(img1).unsqueeze(0), volatile=True).cuda()
                prediction = net(img_var)

                prediction = to_pil(prediction.data.squeeze(0).cpu())
                prediction = prediction.resize((w, h), Image.BILINEAR)
                if args['crf_refine']:
                    prediction = crf_refine(np.array(img), np.array(prediction))
                prediction = np.array(prediction)
                if args['save_results']:
                    Image.fromarray(prediction).save(os.path.join(ckpt_path, exp_name,'(%s) %s_%s' % (exp_name, name, args['snapshot']), img_name + '.png'))





if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end-start)