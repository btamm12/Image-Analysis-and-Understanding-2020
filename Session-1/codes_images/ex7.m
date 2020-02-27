% This Matlab script allows applying convolution filters.

%********************************************************
%*                          INPUT                       *
%********************************************************
close all
clear all

% read image
%filename = 'images/isotropy.bmp'; % TODO 1 : put filename here
filename = 'images/fractal.jpg'; % TODO 1 : put filename here
%filename = 'images/eiffel.jpg'; % TODO 1 : put filename here

im = imread(filename);
if (size(im,3)==3)
    im = rgb2gray(im);
end
f = double(im)/255;
[M,N] = size(f);

% mesh grids
[x,y] = meshgrid([0:1/N:1-1/N],[0:1/M:1-1/M]);
[u,v] = meshgrid([-(N-1)/2:N/2],[-(M-1)/2:M/2]);

%********************************************************
%*                  FILTER COMPUTATION                  *
%********************************************************

 h_x =[ -1 0 1;  % d/dx to use in exercise 7 
      -2 0 2;
      -1 0 1];

 h_y =[ 1   2  1;  % d/dy to use in exercise 7
      0   0  0;
      -1 -2 -1];



%h=h/sum(sum(h));

% filter MTF
H = fft2(h_x,M,N);
H = fftshift(H);
show_func(u,v,abs(H),'MTF');

H = fft2(h_y,M,N);
H = fftshift(H);
show_func(u,v,abs(H),'MTF');

%********************************************************
%*               SPATIAL DOMAIN FILTERING               *
%********************************************************

% filter
f_x = conv2(f,h_x,'same');
f_y = conv2(f,h_y,'same');
f_norm2 = f_x.^2 + f_y.^2; %norm2
f_norm1 = abs(f_x) + abs(f_y); %norm1

h = figure; set(h,'Position',[150   300   1000   500]);
subplot(1,2,1); imagesc(f_norm2); title('filtered image with norm2');
subplot(1,2,2); imagesc(f_norm1); title('filtered image with norm1');
colormap(gray);

