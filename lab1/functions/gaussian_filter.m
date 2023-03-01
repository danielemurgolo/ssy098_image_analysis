function filtered = gaussian_filter(image, std)
% Function Name: gaussian_filter
%
% Description: This function applies a Gaussian filter to a grayscale image
%              with a specified standard deviation. The output is the
%              filtered image. The filter size is at least four standard
%              deviations to maintain precision. The function uses the
%              'fspecial' function to construct a Gaussian filter and the
%              'imfilter' function to apply the filter with the 'symmetric'
%              option.
%
% Inputs:
%   - image : A grayscale image.
%   - std : A number specifying the standard deviation of the Gaussian filter.
%
% Outputs:
%     - filtered : The filtered image.
%
% Example Usage:
%     >> image = imread('example.jpg'); % load an image
%     >> std = 2; % define standard deviation
%     >> filtered = gaussian_filter(image, std);
%     >> imshow(filtered);
%
% Author: Daniele Murgolo
%
% Date: March 1st, 2023

filter = fspecial('gaussian', ceil(4*std), std);

filtered = imfilter(image, filter, 'symmetric');
end
