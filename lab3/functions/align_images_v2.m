function warped = align_images_v2(source, target, threshold, upright)
% Function Name: align_images_v2
%
% Description: This function aligns two input images using SIFT features and RANSAC-based affine
%              transformation estimation.
%
% Inputs:
%   - source: A 2D array representing the source image
%   - target: A 2D array representing the target image
%   - threshold: A scalar value representing the RANSAC distance threshold for inliers
%   - upright: A logical value indicating whether to use upright SIFT features (true) or
%              rotation-invariant SIFT features (false)
%
% Outputs:
%   - warped: A 2D array representing the aligned source image
%
% Example Usage:
%   source = imread('source.png');
%   target = imread('target.png');
%   threshold = 5;
%   upright = true;
%   warped = align_images_v2(source, target, threshold, upright);
%   imshowpair(source, warped, 'montage');
%
% Author: Daniele Murgolo
% Date: March 1st, 2023
%
% See also: detectSIFTFeatures, extractFeatures, matchFeatures, ransac_fit_affine, affine_warp

src_points = detectSIFTFeatures(source);
% src_points = detectSURFFeatures(source);
[src_features, src_validPoints] = extractFeatures(source, src_points, 'Upright', upright);

trg_points = detectSIFTFeatures(target);
% trg_points = detectSURFFeatures(target);
[trg_features, trg_validPoints] = extractFeatures(target, trg_points, 'Upright', upright);

corrs = matchFeatures(src_features, trg_features, 'MaxRatio', 0.8, 'MatchThreshold', 100);

src_points = src_validPoints.Location(corrs(:, 1), :)';

trg_points = trg_validPoints.Location(corrs(:, 2), :)';

[A, t] = ransac_fit_affine(src_points, trg_points, threshold);

warped = affine_warp(size(target), source, A, t);

end