import os
from os import listdir
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from option import  opt
from data_utils import is_image_file
from model import MCNet
import scipy.io as scio  
from eval import PSNR, SSIM, SAM
                   
def main():

    input_path = '/media/hdisk/liqiang/hyperSR/test/' + opt.datasetName + '/' + str(opt.upscale_factor) + '/' 
    out_path = '/media/hdisk/liqiang/hyperSR/result/' +  opt.datasetName + '/' + str(opt.upscale_factor) + '/' 
    
    PSNRs = []
    SSIMs = []
    SAMs = []

    if not os.path.exists(out_path):
        os.makedirs(out_path)
                    
    if opt.cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    model = SSRNet(opt)

    if opt.cuda:
        model = nn.DataParallel(model).cuda()    
        
    checkpoint  = torch.load(opt.model_name)

    model.load_state_dict(checkpoint["model"])       
    images_name = [x for x in listdir(input_path) if is_image_file(x)]           
        
    for index in range(len(images_name)):

        mat = scio.loadmat(input_path + images_name[index]) 
        hyperLR = mat['LR'].transpose(2,0,1).astype(np.float32) 	    	        	
        input = Variable(torch.from_numpy(hyperLR).float(), volatile=True).contiguous().view(1, -1, hyperLR.shape[1], hyperLR.shape[2])    
        if opt.cuda:
           input = input.cuda()                
        output = model(input)  
        HR = mat['HR'].transpose(2,0,1).astype(np.float32)        
        SR = output.cpu().data[0].numpy().astype(np.float32)          
        SR[SR<0] = 0             
        SR[SR>1.] = 1.
        psnr = PSNR(SR, HR)
        ssim = SSIM(SR, HR)
        sam = SAM(SR, HR)
        
        PSNRs.append(psnr)
        SSIMs.append(ssim)
        SAMs.append(sam)
        
        SR = SR.transpose(1,2,0)   
        HR = HR.transpose(1,2,0)  
        
	                    
        scio.savemat(out_path + images_name[index], {'HR': HR, 'SR':SR})  
        print("===The {}-th picture=====PSNR:{:.3f}=====SSIM:{:.4f}=====SAM:{:.3f}====Name:{}".format(index+1,  psnr, ssim, sam, images_name[index],))                 
    print("=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}".format(np.mean(PSNRs), np.mean(SSIMs), np.mean(SAMs))) 
    
if __name__ == "__main__":
    main()
