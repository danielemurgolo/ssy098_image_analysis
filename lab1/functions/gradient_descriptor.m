function desc = gradient_descriptor(image, position, radius)
% Function Name: gradient_descriptor
% 
% Description: This function computes a SIFT-like descriptor at a certain 
%              position. The input radius controls the size of the regions.
% 
% Inputs:
% - image: the input image
% - position: the center position of the descriptor
% - radius: the radius of each square region in the descriptor
% 
% Outputs:
% - desc: the 72-dimensional SIFT-like descriptor vector
% 
% Example Usage:
%     >> img = imread('image.jpg');
%     >> pos = [100, 200];
%     >> r = 5;
%     >> descriptor = gradient_descriptor(img, pos, r);
%     >> disp(descriptor);
% 
% Author: Daniele Murgolo
%
% Date: March 1st, 2023
% 
% See also: gaussian_gradients, place_regions, gradient_histogram
     
std = radius * 0.1;
[grad_x, grad_y] = gaussian_gradients(image, std);
desc = [];

centres = place_regions(position, radius);

for c=centres

    
    x_patch = get_patch(grad_x, c(1), c(2), radius);
    y_patch = get_patch(grad_y, c(1), c(2), radius);
    
    hist = gradient_histogram(x_patch, y_patch);
    
    desc = [desc; hist];

end

desc = desc(:);
desc = desc/norm(desc);

end

