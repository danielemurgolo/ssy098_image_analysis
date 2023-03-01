function histogram = gradient_histogram(grad_x, grad_y)
% Function Name: gradient_histogram
%
% Description:  this function places each gradient into one of eight orientation
%               bins. The function takes in two 1-dimensional arrays of the
%               same size, grad_x and grad_y, representing the gradient
%               magnitudes in the x and y directions, respectively.
%               The provided plot_bouquet lets you plot the histograms as a
%               bouquet of vectors and might be helpful for debugging.
%
% Inputs:
%     - grad_x: a 1-dimensional array representing the gradient magnitudes in the x direction
%     - grad_y: a 1-dimensional array representing the gradient magnitudes in the y direction
%
% Outputs:
%     - histogram: a 1-dimensional array representing the histogram of gradient orientations,
%       with each gradient magnitude placed into one of eight orientation bins
%
% Example Usage:
%     >> grad_x = [1, 2, 3, 4, 5];
%     >> grad_y = [5, 4, 3, 2, 1];
%     >> hist = gradient_histogram(grad_x, grad_y);
%     >> disp(hist);
%
% Author: Daniele Murgolo
%
% Date: March 1st, 2023
%
% See also: atan2, plot_bouquet

bins = 8;
histogram = 1:bins;
angles = atan2(grad_y, grad_x);

for n = 1:bins

    low = (n - 1) * 2 * pi / bins - pi;
    high = n * 2 * pi / bins - pi;

    idx = angles >= low & angles < high;

    gx = grad_x(idx);
    gy = grad_y(idx);

    if isempty(gx)
        gx = 1e-31;
    end

    if isempty(gy)
        gy = 1e-31;
    end

    magnitudes = sqrt(gx.^2+gy.^2);

    histogram(n) = sum(magnitudes);

end


end
