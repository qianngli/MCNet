function [outputArg1,outputArg2] = data_augment(label, upscale_factor, savePath)
    global count
    input = imresize(label, 1/upscale_factor, 'bicubic');       
    count = count+1; 
    count_name = num2str(count, '%05d');
    lr = permute(input, [3 1 2]);
    hr = permute(label, [3 1 2]);
    lr = single(lr);
    hr = single(hr);
    save([savePath,count_name,'.mat'],'lr','hr'); % save augmented hyperspectral image to "savePath"
end

