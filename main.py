import argparse
import os
import os.path as osp
import shutil
import time
import numpy as np
import torch
import torch.optim
import torch.backends.cudnn as cudnn
import gc
from tensorboardX import SummaryWriter
from thop import profile
import pprint
import datetime
import lpips
import pyiqa
import cpbd
import imageio
import sys
from configs.yml_parser import *
from datasets.dataset_sreds import *
from utils import *
from metrics.psnr import *
from metrics.ssim import *
from losses import *
from models.Vgg19 import *
from spikingjelly.clock_driven import functional

os.environ["KMP_BLOCKTIME"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

from models.Spk2ImgMamba import *

parser = argparse.ArgumentParser()
parser.add_argument('--save_name', '-sn', type=str, default='mamba')

parser.add_argument('--data_root', '-dr', type=str, default='/home/jasper/data/REDS120fps') 
parser.add_argument('--arch', '-a', type=str, default='Spk2ImgMamba')
parser.add_argument('--batch_size', '-b', type=int, default=8)
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
parser.add_argument('--configs', '-cfg', type=str, default='./configs/Spk2ImgMamba.yml')
parser.add_argument('--epochs', '-ep', type=int, default=100)
parser.add_argument('--epoch_size', '-es', type=int, default=2400)
parser.add_argument('--workers', '-j', type=int, default=4)
parser.add_argument('--pretrained', '-prt', type=str, default=None)
parser.add_argument('--start_epoch', '-sep', type=int, default=0)
parser.add_argument('--print_freq', '-pf', type=int, default=1)
parser.add_argument('--save_dir', '-sd', type=str, default='ckpt_outputs')
parser.add_argument('--vis_path', '-vp', type=str, default='vis_train')
parser.add_argument('--vis_name', '-vn', type=str, default='Mamba_train')
parser.add_argument('--eval_path', '-evp', type=str, default='vis_eval')
parser.add_argument('--vis_freq', '-vf', type=int, default=2000)
parser.add_argument('--eval', '-e', action='store_true')
parser.add_argument('--w_per', '-wper', type=float, default=0.2)
parser.add_argument('--print_details', '-pd', action='store_true')
parser.add_argument('--milestones', default=[20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70], metavar='N', nargs='*')
parser.add_argument('--lr_scale_factor', '-lrsf', type=float, default=0.7)
parser.add_argument('--eval_interval', '-ei', type=int, default=5)
parser.add_argument('--save_interval', '-si', type=int, default=5)
parser.add_argument('--no_imwrite', action='store_true', default=False)
parser.add_argument('--script_name', type=str, default='Spk2ImgMamba_t2')
args = parser.parse_args()

args.milestones = [int(m) for m in args.milestones]
print('milstones', args.milestones)

cfg_parser = YAMLParser(args.configs)
cfg = cfg_parser.config

cfg['data']['root'] = args.data_root
cfg = add_args_to_cfg(cfg, args, ['batch_size', 'arch', 'learning_rate', 'configs', 'epochs', 'epoch_size', 'workers', 'pretrained', 'start_epoch', 
                        'print_freq', 'save_dir', 'save_name', 'vis_path', 'vis_name', 'eval_path', 'vis_freq', 'w_per', 'script_name'])

n_iter = 0

# load the model at the beginning to prevent reloading
loss_fn_vgg = lpips.LPIPS(net='alex').cuda()


class Logger(object):
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.log.write(message)
        self.terminal.write(message)
        self.log.flush()    #缓冲区的内容及时更新到log文件中
    
    def flush(self):
        pass


def train(cfg, train_loader, model, optimizer, epoch, train_writer):
    ######################################################################
    ## Init
    global n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_name = ['rec_loss', 'per_loss', 'mulscl_loss', 'all_loss']
    losses = AverageMeter(precision=6, i=len(losses_name), names=losses_name)
    model.train()
    torch.cuda.synchronize()
    end = time.time()
    
    vgg19 = Vgg19(requires_grad=False).cuda()
    if torch.cuda.device_count() > 1:
        vgg19 = nn.DataParallel(vgg19, list(range(torch.cuda.device_count())))

    curr_vis_path = os.path.join(cfg['train']['vis_path'], datetime.datetime.now().strftime('%m-%d'))
    make_dir(curr_vis_path)

    # loss_fn_tv2 = VariationLoss(nc=2).cuda()
    # downsampleX2 = nn.AvgPool2d(2, stride=2).cuda()
    loss_fn_L1 = L1Loss()
    
    ######################################################################
    ## Training Loop
    
    for ww, data in enumerate(train_loader, 0):
        
        if ww >= args.epoch_size:
            return 

        spikes = [spk.cuda() for spk in data['spikes']]
        images = [img.cuda() for img in data['images']]
        torch.cuda.synchronize()
        data_time.update(time.time() - end)

        cur_spks = torch.cat(spikes, dim=1)
        
        rec_loss = 0.0
        per_loss = 0.0
        loss_L1_multiscale = 0.0

        seq_len = len(data['spikes']) - 3###corres 23th img GT

        for jj in range(1, 1+seq_len):
            x = cur_spks[:, jj*20-11 : jj*20+50]
            img_gt = images[jj+1]
            # gc.collect()
            # torch.cuda.empty_cache()
            model = model.cuda()
            img_pred = model(x) # output: a sequence of images
            if jj >= 2: 
                rec_loss += loss_fn_L1(img_pred[-1], img_gt, mean=True) / (seq_len - 1)
                if cfg['train']['w_per'] > 0:
                    per_loss += cfg['train']['w_per'] * compute_per_loss_single(img_pred[-1], img_gt, vgg19) / (seq_len - 1)
                else:
                    per_loss = torch.tensor([0.0]).cuda()
                
                pyr_weights = [0.2, 0.5] 
                for l in range(2):
                    loss_L1_multiscale += pyr_weights[l] * loss_fn_L1(img_pred[l], img_gt, mean=True) / (seq_len - 1)

        all_loss = rec_loss + per_loss + loss_L1_multiscale #+ loss_rep_est
        
        # record loss
        losses.update([rec_loss.item(), per_loss.item(), loss_L1_multiscale.item(), all_loss.item()])
        train_writer.add_scalar('rec_loss', rec_loss.item(), n_iter)
        train_writer.add_scalar('per_loss', per_loss.item(), n_iter)
        train_writer.add_scalar('mulscl_loss', loss_L1_multiscale.item(), n_iter)
        train_writer.add_scalar('total_loss', all_loss.item(), n_iter)

        ## compute gradient and optimize
        all_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        functional.reset_net(model)
        
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        torch.cuda.synchronize()
        end = time.time()
        n_iter += 1

        # if n_iter % cfg['train']['vis_freq'] == 0:
        #     vis_img(curr_vis_path, img_pred_0, cfg['train']['vis_name']+'_'+str(n_iter))
        
        if ww % cfg['train']['print_freq'] == 0:
            out_str = 'Epoch: [{:d}] [{:d}/{:d}],  Iter: {:d}  '.format(epoch, ww, len(train_loader), n_iter-1)
            #out_str += 'Time: {},  Data: {}  '.format(batch_time, data_time) #current, average
            out_str += ' '.join(map('{:s} {:.4f} ({:.6f}) '.format, losses.names, losses.val, losses.avg))
            out_str += 'lr {:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
            print(out_str)
        
        torch.cuda.synchronize()
        end = time.time()

    return


def validation(cfg, test_loader, model, epoch, auto_save_path):
    global n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # metrics_name = ['PSNR', 'SSIM', 'LPIPS', 'NIQE', 'BRISQUE', 'CPBD', 'AvgTime']
    metrics_name = ['PSNR', 'SSIM', 'LPIPS', 'NIQE', 'BRISQUE', 'AvgTime']
    all_metrics = AverageMeter(i=len(metrics_name), precision=4, names=metrics_name)

    timestamp1 = datetime.datetime.now().strftime('%m-%d')
    timestamp2 = datetime.datetime.now().strftime('%H%M%S')

    model.eval()

    brisque_loss = pyiqa.create_metric('brisque').cuda()
    niqe_loss = pyiqa.create_metric('niqe').cuda()  
    #lpips_loss = pyiqa.create_metric('lpips').cuda()
    # loss_fn_vgg = lpips.LPIPS(net='alex').cuda() # moved to the start of this program
    
    padder = InputPadder(dims=(720, 1280), padsize=64)

    for ww, data in enumerate(test_loader, 0):
        torch.cuda.synchronize()
        st1 = time.time()
        spikes = torch.cat([spk.cuda() for spk in data['spikes']], dim=1)
        images = data['images']
        torch.cuda.synchronize()
        data_time.update(time.time() - st1)

        seq_metrics = AverageMeter(i=len(metrics_name), precision=4, names=metrics_name)

        seq_len = len(data['spikes']) - 3###corres 23th img GT

        print("seq_len=", seq_len)
        pred_gif=[]
        gt_gif=[]

        for jj in range(1, 1+seq_len):
            x = spikes[:, jj*20-11 : jj*20+50]
            # print("before padding, x.shape", x.shape)
            x = padder.pad(x)[0]
            # print("after padding, x.shape", x.shape)
            
            gt = images[jj+1].cuda()

            if ww==0 and jj==1:
            # calculate the flops
                flops, params = profile(model, (x,))
                print('flops: %.4f G, params: %.4f M' % (flops / 1e9, params / 1e6))
            with torch.no_grad():
                torch.cuda.synchronize()
                st = time.time()
                
                out = model(x) #now a tuple
                torch.cuda.synchronize()
                mtime = time.time() - st
            #rec = torch.clamp(out, 0, 1)
            rec = padder.unpad(out[-1])
            rec_p1 = padder.unpad(out[0])
            rec_p2 = padder.unpad(out[1])

            #rec = spikes[:,jj*20+20:jj*20+21]#save original spike streams

            cur_rec = torch2numpy255(rec)
            cur_rec_p1 = torch2numpy255(rec_p1)
            cur_rec_p2 = torch2numpy255(rec_p2)
            cur_gt = torch2numpy255(gt)

            if not args.no_imwrite and args.eval:
                #save_path = osp.join(args.eval_path, timestamp1, timestamp2)
                save_path = osp.join(args.eval_path, timestamp1)
                make_dir(save_path)
                cur_vis_path1 = osp.join(save_path, '{:03d}_{:03d}_1.png'.format(ww, jj))
                cur_vis_path2 = osp.join(save_path, '{:03d}_{:03d}_2.png'.format(ww, jj))
                cur_vis_path3 = osp.join(save_path, '{:03d}_{:03d}_3.png'.format(ww, jj))

                cv2.imwrite(cur_vis_path1, cur_rec.astype(np.uint8))
                cv2.imwrite(cur_vis_path2, cur_rec_p1.astype(np.uint8))
                cv2.imwrite(cur_vis_path3, cur_rec_p2.astype(np.uint8))

                pred_gif.append(cur_rec.astype(np.uint8))
                gt_gif.append(cur_gt.astype(np.uint8))

            cur_psnr = calculate_psnr(cur_rec, cur_gt)
            cur_ssim = calculate_ssim(cur_rec, cur_gt)
            with torch.no_grad():
                cur_lpips = loss_fn_vgg(rec, gt)

            #lpips_tmp = lpips_loss(cur_gt, cur_rec)
            niqe_tmp = niqe_loss(cur_rec)
            brisque_tmp = brisque_loss(cur_rec)

            # cpbd_tmp = torch.tensor(0., dtype=torch.float32)
            #if args.eval:
                #cpbd_tmp = cpbd.compute(cur_rec)

            # cur_metrics_list = [cur_psnr, cur_ssim, cur_lpips.item(), niqe_tmp.item(), brisque_tmp.item(), cpbd_tmp.item(), mtime]
            cur_metrics_list = [cur_psnr, cur_ssim, cur_lpips.item(), niqe_tmp.item(), brisque_tmp.item(), mtime]
            if args.eval:
                print("[Seq%d, %d-th image]: PSNR:%.4f SSIM:%.4f LPIPS:%.4f NIQE:%.4f BRISQUE:%.4f Time:%.4f" % (ww, jj+2, cur_psnr, cur_ssim, cur_lpips.item(), niqe_tmp.item(), brisque_tmp.item(), mtime))

            all_metrics.update(cur_metrics_list)
            seq_metrics.update(cur_metrics_list)
        
        #if not args.no_imwrite and args.eval:
            #imageio.mimsave(os.path.join(save_path, '{:03d}'.format(ww)+'_duration_'+str(0.1)+'_pred.gif'), pred_gif, duration = 0.1)
            #imageio.mimsave(os.path.join(save_path, '{:03d}'.format(ww)+'_duration_'+str(0.1)+'_gt.gif'),   gt_gif,   duration = 0.1)

        functional.reset_net(model)
            
        if args.print_details:
            print('\n')
            ostr = 'Data{:02d}  '.format(ww) + ' '.join(map('{:s} {:.4f} '.format, seq_metrics.names, seq_metrics.avg))
            print(ostr)
            print()
    
    ostr = 'All  ' + ' '.join(map('{:s} {:.4f} '.format, all_metrics.names, all_metrics.avg))
    print(ostr)

    if args.eval:
        print('\n')
    else:
        print('Test current epoch\n')
        f_metric_avg=open(os.path.join(auto_save_path, 'ckpt_'+args.save_name+'_metric_avg.txt'), 'a+')#Save the files next to the last line
        f_metric_avg.write('%s  ' % (str(epoch).zfill(2)))
        f_metric_avg.write(ostr)
        f_metric_avg.write('GFlops %.6fG ' % (flops / 1e9))
        f_metric_avg.write('\n')
        f_metric_avg.close()

    return

def main():
    ##########################################################################################################
    # Set random seeds
    set_seeds(cfg['seed'])

    # Create save path and logs
    timestamp1 = datetime.datetime.now().strftime('%m-%d')
    timestamp2 = datetime.datetime.now().strftime('%H%M%S')

    if args.save_name == None:
        save_folder_name = 'b{:d}_{:s}'.format(args.batch_size, timestamp2)
    else:
        save_folder_name = 'b{:d}_{:s}_{:s}_{:s}'.format(args.batch_size, args.save_name, timestamp2, cfg['train']['script_name'])

    save_path = osp.join(args.save_dir, timestamp1, save_folder_name)
    print('save path: ', save_path)

    if args.eval:
        print('\n')
    else:
        make_dir(save_path)
        with open(os.path.join(save_path, 'logger.txt'), 'a') as logger_file:
            sys.stdout = Logger(filename=os.path.join(save_path, 'logger.txt'))
        #auto save test results during training
        f_metric_avg=open(os.path.join(save_path, 'ckpt_'+args.save_name+'_metric_avg.txt'), 'w')
        f_metric_avg.close()

    make_dir(args.vis_path)
    make_dir(args.eval_path)
    
    train_writer = SummaryWriter(save_path)

    if args.eval:
        shutil.rmtree(save_path)
        print('remove path: ', save_path)

    cfg_str = pprint.pformat(cfg)
    print('=> configurations: ')
    print(cfg_str)
    
    ##########################################################################################################
    ## Create model
    model = eval(args.arch)()

    if args.pretrained:
        network_data = torch.load(args.pretrained)
        print('=> using pretrained model {:s}'.format(args.pretrained))
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
        model.load_state_dict(network_data, strict=False)
    else:
        network_data = None
        print('=> train from scratch')
        model.init_weights()
        print('=> model params: {:.6f}M'.format(model.num_parameters()/1e6))
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
        '''
        input_data = torch.randint(0, 2, (1, 61, 720, 1280)).unsqueeze(0).float().cuda()
        macs, mdl_params = profile(model.cuda(), (input_data))
        print("Estimated GFLOPs: ", macs / 1e9)
        print('Model Parameters: %.6f MB \n' % (mdl_params / 1e6))
        '''

    cudnn.benchmark = True

    ##########################################################################################################
    ## Create Optimizer
    cfgopt = cfg['optimizer']
    cfgmdl = cfg['model']
    assert(cfgopt['solver'] in ['Adam', 'SGD'])
    print('=> settings {:s} solver'.format(cfgopt['solver']))
    
    param_groups = [{'params': model.parameters(), 'weight_decay': cfgmdl['flow_weight_decay']}]
    if cfgopt['solver'] == 'Adam':
        optimizer = torch.optim.Adam(param_groups, args.learning_rate, betas=(cfgopt['momentum'], cfgopt['beta']))
    elif cfgopt['solver'] == 'SGD':
        optimizer = torch.optim.SGD(param_groups, args.learning_rate, momentum=cfgopt['momentum'])
    
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.0001, threshold_mode='rel', min_lr=1e-5, eps=1e-08)

    ##########################################################################################################
    ## Dataset
    train_set = sreds_train(cfg)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        drop_last=False,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['workers'],
        # pin_memory=True
    )

    test_set = sreds_test(cfg)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        drop_last=False,
        batch_size=1,
        shuffle=False,
        num_workers=cfg['train']['workers']
    )

    ##########################################################################################################
    ## Train or Evaluate
    if args.eval:
        validation(cfg=cfg, test_loader=test_loader, model=model, epoch=0, auto_save_path=save_path)
    else:
        epoch = cfg['train']['start_epoch']
        while(True):
            train(
                cfg=cfg,
                train_loader=train_loader,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_writer=train_writer
            )
            epoch += 1

            # scheduler can be added here
            if epoch in args.milestones:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * args.lr_scale_factor

            # save model
            if epoch % args.save_interval == 0:
                model_save_name = '{:s}_epoch{:03d}.pth'.format(cfg['model']['arch'], epoch)
                torch.save(model.state_dict(), osp.join(save_path, model_save_name))
            
            # if epoch % 5 == 0:
            if epoch % args.eval_interval == 0:
                validation(cfg=cfg, test_loader=test_loader, model=model, epoch=epoch, auto_save_path=save_path)

            if epoch >= cfg['train']['epochs']:
                break

if __name__ == '__main__':
    main()
