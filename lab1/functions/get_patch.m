function patch = get_patch(image, x, y, patch_radius)
% Function Name: get_patch
%
% Description: This function returns a patch from the input image that is
%              centred at (x,y) and extends +/- patch_radius in both x and
%              y directions. If the patch falls outside the borders of the
%              image, the function throws an error. The function works with
%              grayscale and color images.
%
% Inputs:
%   - image : A grayscale or color image.
%   - x : The x-coordinate of the center of the patch.
%   - y : The y-coordinate of the center of the patch.
%   - patch_radius : The range of the patch.
%
% Outputs:
%     - patch : The patch extracted from the image.
%
% Example Usage:
%     >> image = imread('example.jpg'); % load an image
%     >> patch_radius = 2; % define patch radius
%     >> x = 50; % define x coordinate
%     >> y = 50; % define y coordinate
%     >> patch = get_patch(image, x, y, patch_radius);
%     >> imshow(patch);
%
% Author: Daniele Murgolo
%
% Date: March 1st, 2023

x_min = x - patch_radius;
x_max = x + patch_radius;

y_min = y - patch_radius;
y_max = y + patch_radius;

if (x_min <= 0) || (y_min <= 0) || (x_max > size(image, 2)) || (y_max > size(image, 1))

    error('Patch outside image border')

end

patch = image(y_min:y_max, x_min:x_max, :);

end
