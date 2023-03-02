function warped = align_images(source, target, thresh)
% Function Name: align_images
%
% Description: This function takes two grayscale images source and target,
%              along with a threshold value thresh, and aligns the source
%              image to the target image using the Scale-Invariant Feature
%              Transform (SIFT) and a RANSAC-based affine transformation
%              estimator. The function returns the aligned image as output.
%
% Inputs:
%   - source: The grayscale source image to be aligned
%   - target: The grayscale target image used as reference for the alignment
%   - thresh: The threshold used by the RANSAC estimator to distinguish inliers from outliers
%
% Outputs:
%   - warped: The aligned grayscale source image
%
% Example Usage:
% source = imread('source_image.jpg');
% target = imread('target_image.jpg');
% warped = align_images(source, target, 10);
% imshowpair(target, warped, 'montage');
%
% Author: Daniele Murgolo
% Date: March 1st, 2023
%
% See also: detectSIFTFeatures, extractFeatures, matchFeatures, ransac_fit_affine, affine_warp

src_points = detectSIFTFeatures(source);
[src_features, src_validPoints] = extractFeatures(source, src_points);

trg_points = detectSIFTFeatures(target);
[trg_features, trg_validPoints] = extractFeatures(target, trg_points);

corrs = matchFeatures(src_features, trg_features, 'MaxRatio', 0.8, 'MatchThreshold', 100);

src_points = src_validPoints.Location(corrs(:, 1), :)';

trg_points = trg_validPoints.Location(corrs(:, 2), :)';

[A, t] = ransac_fit_affine(src_points, trg_points, thresh);

warped = affine_warp(size(target), source, A, t);


end