function warped = warp_16x16(source)
% Function Name: warp_16x16
%
% Description: This function takes a 16x16 source image and applies an affine
%              transformation to warp it into a new 16x16 output image.
%              The affine transformation is based on a set of 4 control
%              points in the source image and their corresponding locations
%              in the output image. The function uses bilinear interpolation
%              to compute the pixel values of the output image.
%
% Inputs:
%   - source: A 16x16 source image
%
% Outputs:
%   - warped: A 16x16 output image resulting from warping the source image
%
% Example Usage:
%   source = imread('image.jpg');
%   source = imresize(source, [16 16]);
%   warped = warp_16x16(source);
%   imshow(warped);
%
% Author: Daniele Murgolo
% Date: March 1st, 2023
%
% See also: transform_coordinates, sample_image_at, bilinear_interpolation

warped = zeros(16, 16);
for i = 1:16
    for j = 1:16
        pos = [i; j];
        src_yx = transform_coordinates(pos);
        src_yx = flip(src_yx);
        warped(j, i) = sample_image_at(source, src_yx);
    end
end
end