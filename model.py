#coding:utf-8
import torch
import torch.nn as nn
import pdb

class BasicConv3d(nn.Module):
    def __init__(self, wn, in_channel, out_channel, kernel_size, stride, padding=(0,0,0)):
        super(BasicConv3d, self).__init__()
        self.conv = wn(nn.Conv3d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
  
        x = self.conv(x)
        x = self.relu(x)
        return x

class S3Dblock(nn.Module):
    def __init__(self, wn, n_feats):
        super(S3Dblock, self).__init__()

        self.conv = nn.Sequential(
            BasicConv3d(wn, n_feats, n_feats, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            BasicConv3d(wn, n_feats, n_feats, kernel_size=(3,1,1), stride=1, padding=(1,0,0))
        )            
       
    def forward(self, x): 
    	   	
        return self.conv(x)

def _to_4d_tensor(x, depth_stride=None):
    """Converts a 5d tensor to 4d by stackin
    the batch and depth dimensions."""
    x = x.transpose(0, 2)  # swap batch and depth dimensions: NxCxDxHxW => DxCxNxHxW
    if depth_stride:
        x = x[::depth_stride]  # downsample feature maps along depth dimension
    depth = x.size()[0]
    x = x.permute(2, 0, 1, 3, 4)  # DxCxNxHxW => NxDxCxHxW
    x = torch.split(x, 1, dim=0)  # split along batch dimension: NxDxCxHxW => N*[1xDxCxHxW]
    x = torch.cat(x, 1)  # concatenate along depth dimension: N*[1xDxCxHxW] => 1x(N*D)xCxHxW
    x = x.squeeze(0)  # 1x(N*D)xCxHxW => (N*D)xCxHxW
    return x, depth


def _to_5d_tensor(x, depth):
    """Converts a 4d tensor back to 5d by splitting
    the batch dimension to restore the depth dimension."""
    x = torch.split(x, depth)  # (N*D)xCxHxW => N*[DxCxHxW]
    x = torch.stack(x, dim=0)  # re-instate the batch dimension: NxDxCxHxW
    x = x.transpose(1, 2)  # swap back depth and channel dimensions: NxDxCxHxW => NxCxDxHxW
    return x
    
    
class Block(nn.Module):
    def __init__(self, wn, n_feats, n_conv):
        super(Block, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        
        Block1 = []  
        for i in range(n_conv):
            Block1.append(S3Dblock(wn, n_feats)) 
        self.Block1 = nn.Sequential(*Block1)         

        Block2 = []  
        for i in range(n_conv):
            Block2.append(S3Dblock(wn, n_feats)) 
        self.Block2 = nn.Sequential(*Block2) 
        
        Block3 = []  
        for i in range(n_conv):
            Block3.append(S3Dblock(wn, n_feats)) 
        self.Block3 = nn.Sequential(*Block3) 
        
        self.reduceF = BasicConv3d(wn, n_feats*3, n_feats, kernel_size=1, stride=1)                                                            
        self.Conv = S3Dblock(wn, n_feats)
        self.gamma = nn.Parameter(torch.ones(3))   
         
        conv1 = []   
        conv1.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1)))) 
        conv1.append(self.relu)
        conv1.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1))))         
        self.conv1 = nn.Sequential(*conv1)           

        conv2 = []   
        conv2.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1)))) 
        conv2.append(self.relu)
        conv2.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1))))         
        self.conv2 = nn.Sequential(*conv2)  
        
        conv3 = []   
        conv3.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1)))) 
        conv3.append(self.relu)
        conv3.append(wn(nn.Conv2d(n_feats, n_feats, kernel_size=(3,3),  stride=1, padding=(1,1))))         
        self.conv3 = nn.Sequential(*conv3)          
                 
        
                                                          
    def forward(self, x): 
        
        res = x
        x1 = self.Block1(x) + x 
        x2 = self.Block2(x1) + x1         
        x3 = self.Block3(x2) + x2     

        x1, depth = _to_4d_tensor(x1, depth_stride=1)  
        x1 = self.conv1(x1)       
        x1 = _to_5d_tensor(x1, depth)  
                             
        x2, depth = _to_4d_tensor(x2, depth_stride=1)  
        x2 = self.conv2(x2)       
        x2 = _to_5d_tensor(x2, depth)         
   
                     
        x3, depth = _to_4d_tensor(x3, depth_stride=1)  
        x3 = self.conv3(x3)       
        x3 = _to_5d_tensor(x3, depth)  
                
        x = torch.cat([self.gamma[0]*x1, self.gamma[1]*x2, self.gamma[2]*x3], 1)                 
        x = self.reduceF(x) 
        x = self.relu(x)
        x = x + res        
        
        
        x = self.Conv(x)                                                                                                               
        return x  
                                                                                                                        
                        
class SSRNet(nn.Module):
    def __init__(self, args):
        super(SSRNet, self).__init__()
        
        scale = args.upscale_factor
        n_colors = args.n_colors
        n_feats = args.n_feats          
        n_conv = args.n_conv
        kernel_size = 3
        
#        band_mean = (0.0939, 0.0950, 0.0869, 0.0839, 0.0850, 0.0809, 0.0769, 0.0762, 0.0788, 0.0790, 0.0834, 
#                     0.0894, 0.0944, 0.0956, 0.0939, 0.1187, 0.0903, 0.0928, 0.0985, 0.1046, 0.1121, 0.1194, 
#                     0.1240, 0.1256, 0.1259, 0.1272, 0.1291, 0.1300, 0.1352, 0.1428, 0.1541) #CAVE
        band_mean = (0.0100, 0.0137, 0.0219, 0.0285, 0.0376, 0.0424, 0.0512, 0.0651, 0.0694, 0.0723, 0.0816,
                     0.0950, 0.1338, 0.1525, 0.1217, 0.1187, 0.1337, 0.1481, 0.1601, 0.1817, 0.1752, 0.1445, 
                     0.1450, 0.1378, 0.1343, 0.1328, 0.1303, 0.1299, 0.1456, 0.1433, 0.1303) #Hararvd 

#        band_mean = (0.0944, 0.1143, 0.1297, 0.1368, 0.1599, 0.1853, 0.2029, 0.2149, 0.2278, 0.2275, 0.2311,
#                     0.2331, 0.2265, 0.2347, 0.2384, 0.1187, 0.2425, 0.2441, 0.2471, 0.2453, 0.2494, 0.2584,
#                     0.2597, 0.2547, 0.2552, 0.2434, 0.2386, 0.2385, 0.2326, 0.2112, 0.2227) #ICVL

#        band_mean = (0.0483, 0.0400, 0.0363, 0.0373, 0.0425, 0.0520, 0.0559, 0.0539, 0.0568, 0.0564, 0.0591,
#                     0.0678, 0.0797, 0.0927, 0.0986, 0.1086, 0.1086, 0.1015, 0.0994, 0.0947, 0.0980, 0.0973, 
#                     0.0925, 0.0873, 0.0887, 0.0854, 0.0844, 0.0833, 0.0823, 0.0866, 0.1171, 0.1538, 0.1535) #Foster

#        band_mean = (0.0595,	0.0600,	0.0651,	0.0639,	0.0641,	0.0637,	0.0646,	0.0618,	0.0679,	0.0641,	0.0677,
#        	           0.0650,	0.0671,	0.0687,	0.0693,	0.0687,	0.0688,	0.0677,	0.0689,	0.0736,	0.0735,	0.0728,	0.0713,	0.0734,
#        	           0.0726,	0.0722,	0.074,	0.0742,	0.0794,	0.0892,	0.1005) #Foster2002                
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.band_mean = torch.autograd.Variable(torch.FloatTensor(band_mean)).view([1, n_colors, 1, 1])
                                     
        self.head = wn(nn.Conv3d(1, n_feats, kernel_size, padding=kernel_size//2))        
               
        self.SSRM1 = Block(wn, n_feats, n_conv)              
        self.SSRM2 = Block(wn, n_feats, n_conv) 
        self.SSRM3 = Block(wn, n_feats, n_conv)           
        self.SSRM4 = Block(wn, n_feats, n_conv)  
#       self.SSRM5 = Block(wn, n_feats, n_conv)                                                 
        tail = []
        tail.append(wn(nn.ConvTranspose3d(n_feats, n_feats, kernel_size=(3,2+scale,2+scale), stride=(1,scale,scale), padding=(1,1,1))))         
        tail.append(wn(nn.Conv3d(n_feats, 1, kernel_size, padding=kernel_size//2)))  
        self.tail = nn.Sequential(*tail)                                                                                 
               
    def forward(self, x):
    	
        x = x - self.band_mean.cuda()  
        x = x.unsqueeze(1)
        T = self.head(x) 
        
        x = self.SSRM1(T)
        x = torch.add(x, T) 
            
        x = self.SSRM2(x)
        x = torch.add(x, T)   
                           
        x = self.SSRM3(x)
        x = torch.add(x, T)                                

        x = self.SSRM4(x)
        x = torch.add(x, T) 
        
#        x = self.SSRM5(x)
#        x = torch.add(x, T)
                                                                                     
        x = self.tail(x)      
        x = x.squeeze(1)        
        x = x + self.band_mean.cuda()   
        return x                                                   
        
                                             