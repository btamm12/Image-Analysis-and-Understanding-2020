% show_func(x,y,f,fig_title)
%
%       Shows the real part of an image/spectrum in a simple 2d map and a
%       3d surface. No scaling necessary.
%
%       Inputs:
%           x,y : MxN matrix coordinates of function values (use meshgrid)
%           f : MxN matrix with function values (eg image intensities)
%           fig_title : figure title
%
%       Example:
%           Following example will show a (2,-4) cosine image.
%
%           [x,y] = meshgrid([0:0.01:1],[0:0.01:1]);
%           f = cos(2*pi*(x*2-y*4));
%           show_func(x,y,f,'cosine (2,-4) image');
%
function show_func(x,y,f,fig_title)

h = figure;
set(h,'Position',[150   300   1000   500]);

subplot(1,2,1);
pcolor(x,y,real(f)); 
%set(h,'EdgeColor','flat'); 
shading flat;
title([fig_title ' (2d)']);

subplot(1,2,2); h = gca; 
set(h,'ZScale','log');
surf(x,y,real(f),'EdgeColor','flat');
title([fig_title ' (3d)']);