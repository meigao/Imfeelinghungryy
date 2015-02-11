close all
clear
% initialization
run matlab/vl_setupnn

% load model
net = load ('imagenet_vgg_verydeep_16.mat');

% data_path = '../data/';
% data_path = '../testdata/';
data_path = '../restaurants/';
categorylist = dir(data_path);
n = length(categorylist);
for i = 1:n
    if ~categorylist(i).isdir || categorylist(i).name(1) == '.'
        continue;
    end

    fprintf('process restaurant # %d/%d\n', i, n);
    file_path = [data_path, categorylist(i).name,'/'];
        
    if ~exist([file_path, 'data.mat'], 'file');
        % extract vgg feature
        feat = compute_vgg_feat(net, file_path);
        dfeat = double(feat);
        save([file_path, 'data.mat'], 'feat');
        save([file_path, 'data.txt'], 'dfeat','-ascii', '-double', '-tabs');
    end
end

data_path = '../recommends/';

categorylist = dir(data_path);
n = length(categorylist);
for i = 1:n
    if ~categorylist(i).isdir || categorylist(i).name(1) == '.'
        continue;
    end

    fprintf('process restaurant # %d/%d\n', i, n);
    file_path = [data_path, categorylist(i).name,'/'];
    
    if i == 538
        kk = 1;
    end
        
    if exist([file_path, 'name.txt'], 'file');
        fid = fopen([file_path, 'name.txt'], 'r');
        name = fgetl(fid);
        num = str2num(fgetl(fid));
        for k = 1:num
           cate = lower(fgetl(fid));
           if strcmp(cate, 'sushi') == 1 || strcmp(cate, 'janpanese') == 1
               img_list = dir([file_path, '*.jpg']);
               img = imread([file_path, img_list(1).name]);
               imshow(img, [])
           end
        end
        fclose(fid);
    end
end

file_path = '../data/neg/';
feat = compute_vgg_feat(net, file_path);
dfeat = double(feat);
save([file_path, 'data.mat'], 'feat');
save([file_path, 'data.txt'], 'dfeat','-ascii', '-double', '-tabs');