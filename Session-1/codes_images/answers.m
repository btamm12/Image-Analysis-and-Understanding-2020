close all;

% 1. Noise filtering
%
% The Gaussian filter produces a visually more appealing result. The
% downside of using a block filter is that you get "ringing" in the time
% domain. This is due to the convolution with a sinc-like surface in the
% time domain.
%
% The Gaussian filter gives a smoothing effect to the image
%
% Result:
f = openfig('figures/ex1.fig');
set(f, 'NumberTitle', 'off', 'Name', 'Exercise 1');

% 2. Edge detection
%
% Starting from the smoothing Gaussian filter, an edge detector can be
% constructed by calculating (1-H) in the frequency domain. Transforming to
% the spatial domain, we get (delta-h). If the (windowed) image has no
% variations in intensity, the center pixel will get the value 1-h(0,0) and
% the other pixels will be multiplied by the negative Gaussian weights.
% Together, the value will be ~0.
%
% When there is an abrupt edge, there will be a bright spot. Take, for
% example a separation between a white plane (1) and a black plane (0).
%
% At the edge, half of the window will be white and half will be black. The
% center pixel gives +1, the white half gives -0.5 and the black half gives
% 0. Together this gives 0.5.
%
% Just over the edge, the same applies, but the center pixel gives 0. This
% gives -0.5.
%
% ==> This leads to edges where one side is brighter than normal and the
%     other side is darker than normal.
%
% Noise disrupts the edge detector, since it includes high frequencies.
% This can be improved by first smoothing the image with a Gaussian filter.
% The high-frequency noise will be removed, but the edges will still remain
% since their power was so large in the starting image. These smoothed
% edges can still be detected by the edge detector.
%
% Result:
f = openfig('figures/ex2.fig');
set(f, 'NumberTitle', 'off', 'Name', 'Exercise 2');

% 3. Linearity and shift-invariance
%
% (1) Linear
% (2) Linear + shift-invariant
% (3) Shift-invariant

% 4. Median filtering
%
% Median filtering is not linear. A 3x3 median applied twice does not give
% the same result as a 5x5 median applied once. (Try a 1D list).
%
% Median filter of size [4,4] applied 2 times gives decent results. But
% none of the filters look great.
%
% Result:
f = openfig('figures/ex4.fig');
set(f, 'NumberTitle', 'off', 'Name', 'Exercise 4');

% 5. Matched filters
%
%  1  2  0 -2 -1
%  2  4  0 -4 -2
%  0  0  0  0  0
% -2 -4  0  4  2
% -1 -2  0  2  1
%
% Can be separated in [1,2,0,-2,-1]^T and [1,2,0,-2,-1]. Each of these can
% be decomposed in conv([1,0,-1],[1,2,1]). This is a multiplication of a
% vertical/horizontal edge detector with a smoothing filter in the
% frequency domain.
%
% Result:
f = openfig('figures/ex5.fig');
set(f, 'NumberTitle', 'off', 'Name', 'Exercise 5');

% 6. The other way around
%
% H(u,v) = (2 + 2*cos(2*pi*u))*(2 - 2*cos(2*pi*v)) !!! f = 1 in both cos
%        | IFFT
%        V
% h(x,y) = conv( 2*delta(x) + delta(x-1) + delta(x+1),
%                2*delta(y) - delta(y-1) - delta(y+1)  )
%
% h = conv([1,2,1], [-1,2,-1]^T)
% --> see ex5and6and9.m "Example of mask to use in exercise 6"
% --> first = Gaussian smoothing
% --> second = approx. of 2nd order derivative
% --> horizontal edge detector
%
% Result:
f = openfig('figures/ex6.fig');
set(f, 'NumberTitle', 'off', 'Name', 'Exercise 6');

% 7. Isotropy
%
% First operator = 2-norm of gradient
% Second operator = 1-norm of the gradient
%
% Correspondences: both are edge detectors
% Differences: when edges are aligned with x- or y-axis, the two will give
%              the same response, but when the edge is not aligned with the
%              x- or y-axis (e.g. diagonal), then the second one will give
%              a larger response.
%
% Result:
%%
f = openfig('figures/ex7.fig');
set(f, 'NumberTitle', 'off', 'Name', 'Exercise 7');