% This Matlab script allows median filtering.

%********************************************************
%*                          INPUT                       *
%********************************************************
close All
clear all
% read image
filename = 'images/lena.jpg';
im = imread(filename);
if (size(im,3)==3)
    im = rgb2gray(im);
end
f = double(im)/255;
figure; imshow(f); title('input image');
[M,N] = size(f);

%********************************************************
%*                          NOISE                       *
%********************************************************
f = f + 0.05*randn(M,N);


%********************************************************
%*                      MEDIAN FILTERING                *
%********************************************************

% median filter parameters
%    TODO : set correct parameters here
filt_size = 4;  % window size
filt_runs = 2;  % how many times applied consecutively

% filter
f2 = f;
for i=1:filt_runs
    f2 = medfilt2(f2,[filt_size filt_size]);
end

figure;
subplot(1,2,1); imshow(f); title('noisy image');
subplot(1,2,2); imshow(f2); title('median filtered'); 
colormap(gray);

