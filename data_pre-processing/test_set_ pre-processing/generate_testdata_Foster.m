clear 
clc
close all

dataset = 'Foster';
upscale = 4;
savePath = ['H:/test/',dataset,'/',num2str(upscale)];  % save test set  to "savePath"
if ~exist(savePath, 'dir')
    mkdir(savePath)
end

%% obtian all the original hyperspectral image
srPath = 'E:\data\HyperSR\Foster\test';
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
    
    load([srPath,'/',name])
    data = hsi;
    clear hsi

    count = count + 1;
    
    %% normalization
    data = data/(1.0*max(max(max(data))));
    data = data(1:512, 1:512,:);
    
    %% obtian HR and LR hyperspectral image
    HR = modcrop(data, upscale);
    LR = imresize(HR,1/upscale,'bicubic'); %LR  

    save([savePath,'/',name,'.mat'], 'HR', 'LR')

    clear HR
    clear LR

end
