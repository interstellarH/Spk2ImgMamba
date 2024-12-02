import argparse
import os
import os.path as osp
import shutil
import time
import cv2
import numpy as np
import torch
import torch.optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from thop import profile
import pprint
import datetime
import lpips
import pyiqa
import cpbd
import imageio
from configs.yml_parser import *
from datasets.dataset_sreds import *
from utils import *
from metrics.psnr import *
from metrics.ssim import *
from losses import *
from models.Vgg19 import *
from spikingjelly.clock_driven import functional
from datasets.ds_utils import *

# os.environ["KMP_BLOCKTIME"] = "0"
# os.environ["OMP_NUM_THREADS"] = "1"
# torch.set_num_threads(1)

from models.Spk2ImgMamba import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', '-dr', type=str, default='/home/jasper/data/real_spk_data')
parser.add_argument('--arch', '-a', type=str, default='Spk2ImgMamba')
parser.add_argument('--batch_size', '-b', type=int, default=1)
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
parser.add_argument('--configs', '-cfg', type=str, default='./configs/Spk2ImgMamba.yml')
parser.add_argument('--epochs', '-ep', type=int, default=100)
parser.add_argument('--epoch_size', '-es', type=int, default=1000)
parser.add_argument('--workers', '-j', type=int, default=8)
parser.add_argument('--pretrained', '-prt', type=str, default=None)
parser.add_argument('--start_epoch', '-sep', type=int, default=0)
parser.add_argument('--print_freq', '-pf', type=int, default=1)
parser.add_argument('--save_dir', '-sd', type=str, default='ckpt_outputs')
parser.add_argument('--save_name', '-sn', type=str, default='t681')
parser.add_argument('--vis_path', '-vp', type=str, default='vis_train')
parser.add_argument('--vis_name', '-vn', type=str, default='Spk2ImgMamba_train')
parser.add_argument('--eval_path', '-evp', type=str, default='vis_test_real')
parser.add_argument('--vis_freq', '-vf', type=int, default=200)
parser.add_argument('--eval', '-e', action='store_true')
parser.add_argument('--w_per', '-wper', type=float, default=0.2)
parser.add_argument('--print_details', '-pd', action='store_true')
parser.add_argument('--milestones', default=[20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70], metavar='N', nargs='*')
parser.add_argument('--lr_scale_factor', '-lrsf', type=float, default=0.7)
parser.add_argument('--eval_interval', '-ei', type=int, default=5)
parser.add_argument('--save_interval', '-si', type=int, default=5)
parser.add_argument('--no_imwrite', action='store_true', default=False)
args = parser.parse_args()

args.milestones = [int(m) for m in args.milestones]

cfg_parser = YAMLParser(args.configs)
cfg = cfg_parser.config

cfg['data']['root'] = args.data_root
cfg = add_args_to_cfg(cfg, args, ['batch_size', 'arch', 'learning_rate', 'configs', 'epochs', 'epoch_size', 'workers', 'pretrained', 'start_epoch', 
                        'print_freq', 'save_dir', 'save_name', 'vis_path', 'vis_name', 'eval_path', 'vis_freq', 'w_per'])

def gamma_correction(image, gamma):
    normalized = image / 255.0
    
    corrected = np.power(normalized, gamma)
    
    corrected = (corrected * 255).astype(np.float64)
    return corrected
    
n_iter = 0

def validation(cfg, test_path, model, file_name, t_begin, mini_len):
    global n_iter
    data_time = AverageMeter()
    metrics_name = ['NIQE', 'BRISQUE', 'CPBD', 'AvgTime']
    all_metrics = AverageMeter(i=len(metrics_name), precision=4, names=metrics_name)

    timestamp1 = datetime.datetime.now().strftime('%m-%d')
    timestamp2 = datetime.datetime.now().strftime('%H%M%S')

    model.eval()

    brisque_loss = pyiqa.create_metric('brisque').cuda()
    niqe_loss = pyiqa.create_metric('niqe').cuda()  

    padder = InputPadder(dims=(250, 400), padsize=16)

    torch.cuda.synchronize()
    st1 = time.time()
    spikes = np.array(dat_to_spmat(test_path, size=(250, 400)), dtype=np.float32)
    spikes = torch.tensor(spikes).unsqueeze(0).cuda()########
    print(spikes.shape)
    torch.cuda.synchronize()
    data_time.update(time.time() - st1)

    seq_metrics = AverageMeter(i=len(metrics_name), precision=4, names=metrics_name)

    # save_path = osp.join(args.eval_path, timestamp1, file_name)
    save_path = osp.join(args.eval_path, file_name)
    make_dir(save_path)

    seq_len = spikes.size(1) // 20 - 3
    seq_len = min([seq_len, mini_len]) ###默认选择mini_len=30个连续图像进行输出，1K Hz

    pred_gif=[]

    for jj in range(0, seq_len):
        x = spikes[:, jj*20+t_begin : jj*20+t_begin+61]
        
        x = padder.pad(x)[0]
        
        with torch.no_grad():
            torch.cuda.synchronize()
            st = time.time()
            
            out = model(x)

            torch.cuda.synchronize()
            mtime = time.time() - st
        out_0 = torch.clamp(out[-1], 0, 1)
        out_1 = torch.clamp(out[0],0, 1)
        out_2 = torch.clamp(out[1],0, 1)
        rec = padder.unpad(out_0)
        rec_1 = padder.unpad(out_1)
        rec_2 = padder.unpad(out_2)
        
        #rec = spikes[:,jj*20+t_begin+30:jj*20+t_begin+31]#save original spike streams
        
        cur_rec = torch2numpy255(rec)
        cur_rec_1 = torch2numpy255(rec_1)
        cur_rec_2 = torch2numpy255(rec_2)
        
        #cur_rec = gamma_correction(cur_rec, 0.8)#Gamma transformation
        
        if not args.no_imwrite and args.eval:
            
            cur_vis_path1 = osp.join(save_path, 'pred_{:03d}_1.png'.format(jj))
            cur_vis_path2 = osp.join(save_path, 'pred_{:03d}_2.png'.format(jj))
            cur_vis_path3 = osp.join(save_path, 'pred_{:03d}_3.png'.format(jj))
            cv2.imwrite(cur_vis_path1, cur_rec_1.astype(np.uint8))
            cv2.imwrite(cur_vis_path2, cur_rec_2.astype(np.uint8))
            cv2.imwrite(cur_vis_path3, cur_rec.astype(np.uint8))

        pred_gif.append(cur_rec.astype(np.uint8))

        niqe_tmp = niqe_loss(cur_rec)
        brisque_tmp = brisque_loss(cur_rec)

        cpbd_tmp = torch.tensor(0., dtype=torch.float32)
        #if args.eval:
            #cpbd_tmp = cpbd.compute(cur_rec)
        
        if args.eval:
            print("[%d-th process]: NIQE:%.4f BRISQUE:%.4f CPBD:%.4f Time:%.4f" % (jj, niqe_tmp.item(), brisque_tmp.item(), cpbd_tmp.item(), mtime))

        if jj > 0:#不统计第一次的时间
            cur_metrics_list = [niqe_tmp.item(), brisque_tmp.item(), cpbd_tmp.item(), mtime]
            all_metrics.update(cur_metrics_list)
            seq_metrics.update(cur_metrics_list)

    imageio.mimsave(os.path.join(save_path, '00_duration_'+str(0.1)+'_pred.gif'), pred_gif, duration = 0.1)
    
    functional.reset_net(model)
            
    if args.print_details:
        print('\n')
        ostr = ' '.join(map('{:s} {:.4f} '.format, seq_metrics.names, seq_metrics.avg))
        print(ostr)
        print()
    
    ostr = 'All  ' + ' '.join(map('{:s} {:.4f} '.format, all_metrics.names, all_metrics.avg))
    print(ostr)

    return all_metrics.avg
    

def main():
    # make_dir(args.vis_path)
    make_dir(args.eval_path)
    
    ##########################################################################################################
    ## Create model
    model = eval(args.arch)()

    if args.pretrained:
        network_data = torch.load(args.pretrained)
        print('=> using pretrained model {:s}'.format(args.pretrained))
        # model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()
        model.load_state_dict(network_data, strict=False)

    cudnn.benchmark = True
    
    metrics_name = ['NIQE', 'BRISQUE', 'CPBD', 'AvgTime']
    metrics_real = AverageMeter(i=len(metrics_name), precision=4, names=metrics_name)
    ##########################################################################################################
    ## Dataset
    minimal_len = [30, 30, 30, 30]
    file_path_all = ['PKU-Spike-High-Speed/Class_A',  'recVidarReal2019/classA', 'recVidarReal2019/classB', 'momVidarReal2021/data']
    # file_path_all = ['PKU-Spike-High-Speed/Class_A']
    file_name_all = []
    file_name_all.append(['bus', 'car-100kmh', 'rotation1-2600rpm', 'rotation2-2600rpm'])
    file_name_all.append(['ballon', 'car-100kmh', 'rotation1', 'rotation2', 'rotation2x'])
    file_name_all.append(['forest', 'railway', 'train-350kmh', 'viaduct-bridge'])
    file_name_all.append(['Apple_1', 'Basketball_1', 'Basketball_2', 'Bottle_1', 'Clock_1', 'Clock_2', 'DroneAndBalls_1', 'FlyingBalls', 'Fruits_1', 'Football_1', 'Football_2', 'Keyboard_1', 'Scissors_1', 'Tennisball_1'])#, \
                          #'Badminton_HitNet1', 'Badminton_HitNet2', 'Badminton_HitNet3', 'Badminton_HitNet4', 'Badminton_Sideline1', 'Badminton_Sideline2', 'Basketball_2', 'DroneAndBalls_2', 'DroneAndBalls_3', 'DroneAndBalls_4', \
                          #'FlyingPoker_3', 'FlyingPokerPingpong', 'Flying-Rotation', 'Flying-Rotation-Ego', 'Badminton_Sideline2', 'Keyboard_2', 'LaserPoint_1', 'LaserPoint_5-1', 'LifeScene_1-1', 'LifeScene_1-2', 'Mouse_1', 'Pingpong_Serve', 'Tennisball_2'])

    for ii in range(len(file_path_all)):
        file_path = file_path_all[ii]

        for jj in range(len(file_name_all[ii])):
            name = file_name_all[ii][jj]

            test_path = osp.join(args.data_root, file_path, name+'.dat')
            print(test_path)

            parent_directory = os.path.basename(os.path.dirname(os.path.dirname(test_path)))
            folder_name = os.path.basename(os.path.dirname(test_path))
            file_name = os.path.splitext(os.path.basename(test_path))[0]
            extracted_path_name = osp.join(parent_directory, folder_name, file_name)

            with torch.no_grad():
                cur_metrics_avg = validation(cfg=cfg, test_path=test_path, model=model, file_name=extracted_path_name, t_begin=19, mini_len=minimal_len[ii])
                if ii==1 or ii==2:
                    metrics_real.update(cur_metrics_avg)
            print('\n')

    print('recVidarReal2019  ' + ' '.join(map('{:s} {:.4f} '.format, metrics_real.names, metrics_real.avg)) + '\n')

    file_path = 'recVidarReal2019/classA'   #['PKU-Spike-High-Speed/Class_A', 'PKU-Spike-High-Speed/Class_B', 'recVidarReal2019/classA', 'recVidarReal2019/classB', 'momVidarReal2021/data']

    #name = 'car-100kmh' #'bus'   'car-100kmh'   'rotation1-2600rpm'   'rotation2-2600rpm'
    #name = 'viaduct-bridge'    #'viaduct-bridge'   'train-350kmh'   'railway'   'forest'

    name = 'rotation1'    #'ballon'   'car-100kmh'   'rotation1'   'rotation2'   'rotation2x'
    # name = 'viaduct-bridge'    #'viaduct-bridge'   'train-350kmh'   'railway'   'forest'

    #name = 'Apple_1'    #'Clock_1'   'Cup_1'   'Keyboard_1'   'Apple_1'   'FlyingBalls'   'Spoon_1'   'Basketball_1'   'Football_1'   'Tennisball_1'

    test_path = osp.join(args.data_root, file_path, name+'.dat')
    print(test_path)
    
    ##########################################################################################################
    parent_directory = os.path.basename(os.path.dirname(os.path.dirname(test_path)))
    folder_name = os.path.basename(os.path.dirname(test_path))
    file_name = os.path.splitext(os.path.basename(test_path))[0]
    extracted_path_name = osp.join(parent_directory, folder_name, file_name)
    #extracted_file_name = f"{folder_name}_{file_name}"

    ## Evaluate
    validation(cfg=cfg, test_path=test_path, model=model, file_name=extracted_path_name, t_begin=19, mini_len=22)
    

if __name__ == '__main__':
    main()
