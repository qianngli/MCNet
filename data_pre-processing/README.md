**The floder contains three parts: training set augment, test set pre-processing, and band mean.**
-------------------------

**training set augment**
-------------------------

After obtaining training set, we augmet these original hyperspectral image by training_set_augment floder. 
You can  run each .m file in MARLAB arrocding to the corresponding dataset.
 
Note that 'savePath' and 'srPath' need to be redefined according to the location of the file.
Moreover, 'upscale_factor' also can be changed. 

**test set pre-processing**
----------------------------
The floder is to reconstruct test set. Simialrity, you can run .m file in MARLAB arrocding to the corresponding dataset.


**band mean**
---------------------------
The floder is to compute the mean  of each band with respect to traning set. You can run bandmean.m file in MARLAB 

