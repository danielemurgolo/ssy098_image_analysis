function [grad_x,grad_y] = gaussian_gradients(image,std)
% Function Name: gaussian_gradients
%
% Description: This function estimates both the Gaussian derivatives for
%              each pixel in a grayscale image. The function uses filtering 
%              with derivative filters and the 'gaussian_filter' function. 
%              The output is two matrices of the same size as the input 
%              image, representing the estimated Gaussian gradients in the
%              x and y directions.
%
% Inputs:
%   - image : A grayscale image.
%   - std : A number specifying the standard deviation of the Gaussian filter.
%
% Outputs:
%     - grad_x : The estimated Gaussian gradients in the x direction.
%     - grad_y : The estimated Gaussian gradients in the y direction.
%
% Example Usage:
%     >> image = imread('example.jpg'); % load an image
%     >> std = 2; % define standard deviation
%     >> [grad_x, grad_y] = gaussian_gradients(image, std);
%     >> imshow(grad_x);
%     >> imshow(grad_y);
%
% Author: Daniele Murgolo
%
% Date: March 1st, 2023

filtered = gaussian_filter(image, std);
grad_x = imfilter(filtered, [-0.5, 0, 0.5], 'conv');
grad_y = imfilter(filtered, [-0.5; 0; 0.5], 'conv');
end

