function [grad_x,grad_y] = gaussian_gradients(image,std)
%GAUSSIAN_GRADIENTS Summary of this function goes here
%   Detailed explanation goes here
filtered = gaussian_filter(image, std);
grad_x = imfilter(filtered, [-0.5, 0, 0.5], 'conv');
grad_y = imfilter(filtered, [-0.5; 0; 0.5], 'conv');
end

