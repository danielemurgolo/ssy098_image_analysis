function warped = align_images_inlier_ls(source, target, threshold, upright)
% Function Name: align_images_inlier_ls
%
% Description: Align two images by detecting and matching features and applying
%              an affine transformation on inlier matches using the least
%              squares method.
%
% Inputs:
%   - source: the source image
%   - target: the target image
%   - threshold: the threshold used for RANSAC and residual length calculations
%   - upright: whether to use upright SIFT features or not
%
% Outputs:
%   - warped: the source image warped to match the target image
%
% Example usage:
%   % Load two images and call the function with the source and target images,
%   % a threshold value, and a boolean flag indicating whether to use upright
%   % features or not:
%   source_img = imread('source_image.jpg');
%   target_img = imread('target_image.jpg');
%   threshold = 10;
%   upright = false;
%   aligned_img = align_images_inlier_ls(source_img, target_img, threshold, upright);
%
% Author: Daniele Murgolo
% Date: March 1st, 2023

src_points = detectSIFTFeatures(source);
% src_points = detectSURFFeatures(source);
[src_features, src_validPoints] = extractFeatures(source, src_points, 'Upright', upright);

trg_points = detectSIFTFeatures(target);
% trg_points = detectSURFFeatures(target);
[trg_features, trg_validPoints] = extractFeatures(target, trg_points, 'Upright', upright);

corrs = matchFeatures(src_features, trg_features);

src_points = src_validPoints.Location(corrs(:, 1), :)';

trg_points = trg_validPoints.Location(corrs(:, 2), :)';

[A, t] = ransac_fit_affine(src_points, trg_points, threshold);

res = residual_lgths(A, t, src_points, trg_points);

idx = res < threshold;

disp('All points')

disp(num2str(sum(res)))

[A, t] = least_squares_affine(src_points(:, idx), trg_points(:, idx));

res = residual_lgths(A, t, src_points(:, idx), trg_points(:, idx));

disp('Only inliers')

disp(num2str(sum(res)))


warped = affine_warp(size(target), source, A, t);
end