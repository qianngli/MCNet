clear 
clc
close all

dataset = 'CAVE';
upscale = 2;
savePath = ['H:/test/',dataset,'/',num2str(upscale)];  % save test set  to "savePath"
if ~exist(savePath, 'dir')
    mkdir(savePath)
end

%% obtian all the original hyperspectral image
srPath = 'E:\data\HyperSR\CAVE\test';
srFile=fullfile(srPath);
srdirOutput=dir(fullfile(srFile));
srfileNames={srdirOutput.name}';
number = length(srfileNames)

for index = 1 : number
    name = char(srfileNames(index));
    if(isequal(name,'.')||... % remove the two hidden folders that come with the system
           isequal(name,'..'))
               continue;
    end
    disp(['-----deal with:',num2str(index),'----name:',name]);     
    
    singlePath= [srPath,'\', name, '\', name];
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
    imgz=imgz./65535;
    img=reshape(imgz,width*height, Band);

    %% obtian HR and LR hyperspectral image
    hrImage = reshape(img, width, height, Band);
    
    HR = modcrop(hrImage, upscale);
    LR = imresize(HR,1/upscale,'bicubic'); %LR  
    save([savePath,'/',name,'.mat'], 'HR', 'LR')

    clear source
    clear HR
    clear LR
end
