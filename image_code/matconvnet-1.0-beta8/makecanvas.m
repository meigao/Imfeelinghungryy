% path = '../data/pizza/';
% img_list = dir([path,'*.jpg']);
% canva = zeros(800, 2000,3,'uint8');
% 
% idx = 1;
% for i = 1:20
%    for j = 1:8
%       img = imread([path, img_list(idx).name]);
%       img = imresize(img, [100,100],'bilinear');
%       idx = idx +1;
%       canva(1 + (j-1)*100:j*100, 1 + (i-1)*100:i*100,:) = img;
%    end
% end
% imwrite(canva, '2.png','png')

categories = struct('name', 'aa');
path = '../data/';
dir_list = dir(path);

% search all sub-directories for each category
idx = 1;
for i = 1:length(dir_list)
    if ~dir_list(i).isdir || dir_list(i).name(1) == '.'
        continue;
    end
    categories(idx).name = dir_list(i).name;
    idx = idx + 1;
end

n = length(categories);
k = randsample(n, 160, true);

canva2 = zeros(800, 2000, 3, 'uint8');

for i = 1:20
   for j = 1:8
      cate = categories(k(i)).name;
      img_path = [path, cate, '/'];
      img_list = dir([img_path, '*.jpg']);
      m = randsample(length(img_list), 1);
      img = imread([img_path, img_list(m).name]);
      img = imresize(img, [100,100],'bilinear');
      canva2(1 + (j-1)*100:j*100, 1 + (i-1)*100:i*100,:) = img;
   end
end
imwrite(canva2, 'canva2.png','png')
