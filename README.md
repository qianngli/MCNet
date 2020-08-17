MCNet
======
**This is an implementation of  Mixed  2D/3D Convolutional Network for Hyperspectral Image Super-Resolution.**

Dataset
------
**Three public datasets, i.e., [CAVE](https://www1.cs.columbia.edu/CAVE/databases/multispectral/ "CAVE"), [Harvard](http://vision.seas.harvard.edu/hyperspec/explore.html "Harvard"), [Foster](https://personalpages.manchester.ac.uk/staff/d.h.foster/Local\_Illumination\_HSIs/Local\_Illumination\_HSIs\_2015.html "Foster"), are employed to verify the effectiveness of the  proposed MCNet. Since there are too few images in these datasets for deep learning algorithm, we augment the training data. With respect to the specific details, please see the implementation details section.**

**Moreover, we also provide the code about data pre-processing in floder "data pre-processing". The floder contains three parts, including training set augment, test set pre-processing, and band mean for all training set.**

Requirement
---------
**python 2.7, Pytorch 0.3.1, cuda 9.0**

Training
--------
**The ADAM optimizer with beta_1 = 0.9, beta _2 = 0.999 is employed to train our network.  The learning rate is initialized as 10^-4 for all layers, which decreases by a half at every 35 epochs.**

**You can train or test directly from the command line as such:**

###### # python train.py --cuda --datasetName CAVE  --upscale_factor 4
###### # python test.py --cuda --model_name checkpoint/model_4_epoch_200.pth

Result
--------
**To qualitatively measure the proposed MCNet, three evaluation methods are employed to verify the effectiveness of the algorithm, including  peak signal-to-noise ratio (PSNR), structural similarity (SSIM), and spectral angle mapping (SAM).**


| Scale  |  CAVE |  Harvard |  Foster |
| :------------: | :------------: | :------------: | :------------: | 
|  x2 |  45.102 / 0.9738 / 2.241 | 46.263 / 0.9827 / 1.883  | 58.878 / 0.9988 / 4.061 | 
|  x3 |  41.031 / 0.9526 / 2.809  |  42.681 / 0.9627 / 2.214 | 55.017 / 0.9970 / 5.126  |   
|  x4 | 39.026 / 0.9319 / 3.292 |  40.081 / 0.9367 / 2.410 | 52.225 / 0.9941 / 5.685  | 

![Image text](https://github.com/qianngli/Images/blob/master/MCNet/visual_cave_lemons.png)

Citation 
--------
**Please consider cite MCNet for hyperspectral image super-resolution if you find it helpful.**

@article{Li2020mixed,

	title={Mixed  {2D/3D} Convolutional Network  for Hyperspectral Image Super-Resolution},
	author={Q. Li and Q. Wang and X. Li},
	journal={Remote Sensing},
	volume={12},
	number={10},
	pages={1660},
	year={2020}}
  
--------
If you has any questions, please send e-mail to liqmges@gmail.com.
