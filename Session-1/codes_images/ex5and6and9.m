% This Matlab script allows applying convolution filters.

%********************************************************
%*                          INPUT                       *
%********************************************************
close all
clear all
% read image
filename = 'images/skyscraper.jpg'; % TODO 1 : put filename here
%filename = 'images/cosine.jpg'; % TODO 1 : put filename here
%filename = 'images/blury_lena.jpg'; % TODO 1 : put filename here

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

% convolution mask
 h = [  1,  2,  0, -2, -1;
        2,  4,  0, -4, -2;
        0,  0,  0,  0,  0;
       -2, -4,  0,  4,  2;
       -1, -2,  0,  2,  1]; % TODO 2 : enter mask here

   
 h = [ -1 -2 -1;  %Example of mask to use in exercise 6 (this responds to horizontal lines)
       2  4  2;
      -1 -2 -1];
   


  %h =[ -1 0  1]; %Example of mask to use in exercise 8 


  %Exercise 9
%  C=15;  % The constant C must be chosen to regulate the amount of reversed diffusion
%   h =[-1  -2   -1; %Example of mask to use in exercise 9
%       -2  C+12 -2;
%       -1  -2   -1];




%h=h/sum(sum(h));

% filter MTF
H = fft2(h,M,N);
H = fftshift(H);
show_func(u,v,abs(H),'MTF');

%********************************************************
%*               SPATIAL DOMAIN FILTERING               *
%********************************************************

% filter
f2 = conv2(f,h,'same');

h = figure; set(h,'Position',[150   300   1000   500]);
subplot(1,2,1); imagesc(f); title('input image');
subplot(1,2,2); imagesc(f2); title('filtered image');
colormap(gray);

