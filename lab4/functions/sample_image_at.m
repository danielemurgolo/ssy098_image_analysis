function value = sample_image_at(img, position) 
% Function Name:
%     sample_image_at
%
% Description:
%     This function takes an RGB image and a 2x1 vector position, and
%     returns the corresponding RGB value to the position. If the
%     coordinates are outside the image or NaN, the function returns color
%     black.
%
% Inputs:
%     - img        : An RGB image.
%     - position   : A 2 x 1 vector representing the pixel position in the
%                  image.
%
% Outputs:
%     - value      : A 3 x 1 vector representing the RGB value of the pixel
%                  at the specified position in the image.
%
% Example Usage:
%     >> img = imread(...);  % Load an image
%     >> position = [...]; % Define (y,x) coordinates
%     >> value = sample_image_at(img, position)
%     >> disp(value);
%
% Author: Daniele Murgolo
% Date: March 1st, 2023
y = round(position(1));
x = round(position(2));
h = size(img,1);
w = size(img,2);
if(y<=0) || (y>h) || isnan(y)
    value = [0;0;0];
elseif (x<=0) || (x>w) || isnan(x)
    value = [0;0;0];
else
    value = squeeze(img(y,x,:));
end