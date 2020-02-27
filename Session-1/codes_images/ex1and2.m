% This Matlab script allows comparison between a gaussian
% and rectangular noise filter.

%********************************************************
%*                          INPUT                       *
%********************************************************
close All
clear All

% read image
filename = 'images/lena.jpg';
im = imread(filename);
if (size(im,3)==3)
    im = rgb2gray(im);
end
f = double(im)/255;
figure; imshow(f); title('input image');
[M,N] = size(f);

% mesh grids
[x,y] = meshgrid([0:1/N:1-1/N],[0:1/M:1-1/M]);
[u,v] = meshgrid([-(N-1)/2:N/2],[-(M-1)/2:M/2]);

F = fft2(f);
F = fftshift(F);
show_func(u,v,log10(abs(F)+1),'spectrum of input image');


%********************************************************
%*                          NOISE                       *
%********************************************************
f = f + 0.05*randn(M,N);
figure; imshow(f); title('noisy image (We added random noise)'); %image with added noise

% get centered spectrum
F = fft2(f);
F = fftshift(F);
show_func(u,v,log10(abs(F)+1),'spectrum of noisy image');

%********************************************************
%*                  FILTER COMPUTATION                  *
%********************************************************

% requirements
f_max = N/8;

% disk rectangular filter
H_rect = zeros(M,M);
tmp = fspecial('disk',f_max);  %creates a filter (average)
tmp = tmp/max(max(tmp));
mid = M/2 + [-(size(tmp,1)-1)/2:(size(tmp,1)-1)/2];
H_rect(mid,mid) = tmp;
Hu_rect = zeros(M);
Hu_rect(mid) = 1;

% separable gaussian filter
sigma_f = f_max/1.5;
%sigma_f=10

Hu_gauss = normpdf([-N/2:(N-1)/2],0,sigma_f);
Hu_gauss = Hu_gauss/max(max(Hu_gauss));

% corresponding spatial convolution masks (1d)
hu_rect = real(fftshift(ifft(ifftshift(Hu_rect))));
hu_gauss = real(fftshift(ifft(ifftshift(Hu_gauss))));


figure; 
subplot(1,2,1); plot(u(1,:),Hu_rect); title('rect filter');
subplot(1,2,2); plot(u(1,:),Hu_gauss); title('gauss filter');

figure; 
subplot(1,2,1); plot(x(1,:),hu_rect); title('rect filter conv mask');
subplot(1,2,2); plot(x(1,:),hu_gauss); title('gauss filter conv mask');


% 2D Filters
H_gauss = Hu_gauss'*Hu_gauss;


%********************************************************
%*             FREQUENCY DOMAIN FILTERING               *
%********************************************************

% filter
F_rect = F.*H_rect;
F_gauss = F.*H_gauss;

show_func(u,v,log10(abs(F_rect)+1),'rectangular filtered spectrum');
show_func(u,v,log10(abs(F_gauss)+1),'gaussian filtered spectrum');

% return to spatial domain;
%F_rect = ifftshift(F_rect);
%F_gauss = ifftshift(F_gauss);
f_rect = real(ifft2(ifftshift(F_rect)));
f_gauss = real(ifft2(ifftshift(F_gauss)));

h = figure; set(h,'Position',[150   300   1000   500]);
subplot(1,2,1); imshow(f_rect); title('filtered with rectangular');
subplot(1,2,2); imshow(f_gauss); title('filtered with gaussian');
%colormap(gray);
%%

%%%% EDGE DETECTOR

% separable gaussian filter
%sigma_f = f_max/1.5;
sigma_f=10;

Hu_gauss = normpdf([-N/2:(N-1)/2],0,sigma_f);
Hu_gauss = Hu_gauss/max(max(Hu_gauss));
H_gauss = Hu_gauss'*Hu_gauss; imshow(f); title('input image');

% Separable edge detector
%   TODO (for exercise 2: edge detection): alter next code line to detect edges
H_edge=1-H_gauss;
f_edge=real(ifft2(ifftshift(F.*H_edge)));

Hu_edge = H_edge(floor(size(H_edge,1)/2),:);
hu_edge = real(fftshift(ifft(ifftshift(Hu_edge))));

% Edge filter the smoothed image
f_gauss_edge = real(ifft2(ifftshift(F_gauss.*H_edge)));


h = figure; set(h,'Position',[150   300   1000   500]);
subplot(3,3,[1,4,7]); plot(u(1,:),Hu_edge); title('edge filter');
subplot(3,3,[2,5,8]); plot(x(1,:),hu_edge); title('edge filter conv mask');
subplot(3,3,3); imshow(f); title('input image');
subplot(3,3,6); imshow(f_edge, []); title('edge detector');
subplot(3,3,9); imshow(f_gauss_edge, []); title('smoothing + edge detector');

