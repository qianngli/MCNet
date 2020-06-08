clc
clear 
close all
 
%% define hyperparameters 
patchSize = 32;
randomNumber = 32;
upscale_factor = 4;
data_type = 'Foster';
P = 0.5;
global count
count = 0;
imagePatch = patchSize*upscale_factor;
scales = [1.0, 0.75, 0.5];

%% bulid upscale folder
savePath=['G:/',data_type,'/',num2str(upscale_factor),'/'];
if ~exist(savePath, 'dir')
    mkdir(savePath)
end

%% 
src_path =  'E:/data/HyperSR/Foster/train/';
fileFolder=fullfile(src_path);
dirOutput=dir(fullfile(fileFolder,'*.mat'));
fileNames={dirOutput.name}';
length(fileNames)


for i = 1:length(fileNames)
 name = char(fileNames(i));
 disp(['----:',data_type,'----upscale_factor:',num2str(upscale_factor),'----deal with:',num2str(i),'----name:',name]);
 data_path = [src_path, '/', name];
 load(data_path)
 
%% normalization
 radHeight = size(hsi, 1);
 radWidth = size(hsi, 2);
 t = hsi / (1.0*max(max(max(hsi))));
 clear hsi;

%%
for j = 1:randomNumber 
    for sc = 1:length(scales)
        newt = imresize(t, scales(sc));  

        x_random = randperm(size(newt,1) - imagePatch, randomNumber);
        y_random = randperm(size(newt,2) - imagePatch, randomNumber);
        hrImage = newt(x_random(j):x_random(j)+imagePatch-1, y_random(j):y_random(j)+imagePatch-1, :);

        label = hrImage;   
        data_augment(label, upscale_factor, savePath);
        if (rand(1)> P)
            label = imrotate(hrImage,180);  
            data_augment(label, upscale_factor, savePath);
        end
        if (rand(1)> P)
            label = imrotate(hrImage,90);  
            data_augment(label, upscale_factor, savePath);
        end
         if (rand(1)> P)
            label = imrotate(hrImage,270); 
            data_augment(label, upscale_factor, savePath);
         end
        if (rand(1)> P)
            label = flipdim(hrImage,1); 
            data_augment(label, upscale_factor, savePath);
        end
        if (rand(1)> P)
            label = flipdim(hrImage,2);  
            data_augment(label, upscale_factor, savePath);
        end    

        clear x_random;
        clear y_random;
        clear newt;
    end
end
clear t;
end




