function value = sample_image_at(img, position)
% Function Name: sample_image_at
%
% Description: This function takes an image and a position in the form of a
%              2D vector as input, and returns the pixel value at the given
%              position. If the position is out of bounds (i.e., beyond the
%              width or height of the image), the function returns a value
%              of 1.
%
% Inputs:
%   -img: The input image, as an HxW matrix
%   -position: The position of the pixel to sample, as a 2D vector in the form [y, x]
%
% Outputs:
%   -value: The pixel value at the given position, or 1 if the position is out of bounds
%
% Example usage:
%   img = imread('example_image.jpg');
%   position = [100, 200];
%   value = sample_image_at(img, position);
%   disp(value);
%
% Author: Daniele Murgolo
% Date: March 1st, 2023

y = round(position(1));
x = round(position(2));
h = size(img, 1);
w = size(img, 2);
if (y <= 0) || (y > h)
    value = 1;
elseif (x <= 0) || (x > w)
    value = 1;
else
    value = img(y, x);
end