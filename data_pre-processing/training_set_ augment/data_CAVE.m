clc
clear 
close all
 
%% define hyperparameters 
Band = 31;  
patchSize = 32;
randomNumber = 24;
upscale_factor = 2;
data_type = 'CAVE';
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
srPath = 'E:/datas/HyperSR/CAVE/train/';  %source data downlaoded from website 
srFile=fullfile(srPath);
srdirOutput=dir(fullfile(srFile));
srfileNames={srdirOutput.name}';
number = length(srfileNames)-2


for index = 1:length(srfileNames)
    name = char(srfileNames(index));
    if(isequal(name,'.')||... % remove the two hidden folders that come with the system
           isequal(name,'..'))
               continue;
    end
    disp(['----:',data_type,'----upscale_factor:',num2str(upscale_factor),'----deal with:',num2str(index-2),'----name:',name]);

    singlePath= [srPath, name, '/', name];
    singleFile=fullfile(singlePath);
    srdirOutput=dir(fullfile(singleFile,'/*.png'));
    singlefileNames={srdirOutput.name}';
    Band = length(singlefileNames);
    source = zeros(512*512, Band);
    for i = 1:Band
        srName = char(singlefileNames(i));
        srImage = imread([singlePath,'/',srName]);
        if i == 1
            width = size(srImage,1);
            height = size(srImage,2);
        end
        source(:,i) = srImage(:);   
    end

    %% normalization
    imgz=double(source(:));
    img=imgz./65535;
    t = reshape(img, width, height, Band);

    %%
    for sc = 1:length(scales)
        newt = imresize(t, scales(sc));    
        x_random = randperm(size(newt,1) - imagePatch, randomNumber);
        y_random = randperm(size(newt,2) - imagePatch, randomNumber);

        for j = 1:randomNumber
            hrImage = newt(x_random(j):x_random(j)+imagePatch-1, y_random(j):y_random(j)+imagePatch-1, :);

            label = hrImage;   
            data_augment(label, upscale_factor, savePath);

            label = imrotate(hrImage,180);  
            data_augment(label, upscale_factor, savePath);

            label = imrotate(hrImage,90);
            data_augment(label, upscale_factor, savePath);

            label = imrotate(hrImage,270);
            data_augment(label, upscale_factor, savePath);

            label = flipdim(hrImage,1);
            data_augment(label, upscale_factor, savePath);

        end
        clear x_random;
        clear y_random;
        clear newt;

    end
    clear t;
end