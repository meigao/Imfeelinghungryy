function feat = compute_vgg_feat(net, file_path)
% compute 4096 feature

% find image from a directory
imglist = dir([file_path, '*.jpg']);

% if not jpg try to load png images
if isempty(imglist)
    imglist = dir([file_path, '*.png']);
end

% # of images
n = length(imglist);

% pre-allocate space to store features
feat = zeros(n, 4096, 'single');

for i = 1:n
    fprintf(['process ', file_path, ' # of image %d\n'], i);
    % read image from img_list and convert it to color image with float
    % precision
    im = color(single(imread([file_path, imglist(i).name])));
    % resize the original image to 224 * 224 pixels so that we can extract features
    % using vgg network
    % the original image size could be arbitrary size
    im = imresize(im, net.normalization.imageSize(1:2));
    % subtract mean of image intensity for normalization
    im = im - net.normalization.averageImage;
    % perform convolution using vgg
    res = vl_simplenn(net, im);
    % extract 4096 feature from the last 3rd layer and store it into feat
    feat(i,:) = squeeze(res(end-2).x);
end