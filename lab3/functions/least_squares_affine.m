function [A, t] = least_squares_affine(pts, pts_tilde)
% Function Name: least_squares_affine
%
% Description: This function takes a set of input points pts and their
%              transformed counterparts pts_tilde, and estimates the affine
%              transformation that maps pts to pts_tilde in the least squares
%              sense. The estimated affine transformation A and the translation
%              vector t are returned.
%
% Inputs:
%   - pts: A 2xN matrix containing N input points
%   - pts_tilde: A 2xN matrix containing N transformed points corresponding to the input points
%
% Outputs:
%   - A: The estimated 2x2 affine transformation matrix
%   - t: The estimated 2x1 translation vector
%
%
% Example Usage:
% [pts, pts_tilde, A_true, t_true] = affine_test_case();
% [A_init, t_init] = estimate_affine(pts, pts_tilde);
% inliers = ransac_affine(pts, pts_tilde);
% [A_refined, t_refined] = least_squares_affine(pts(:, inliers), pts_tilde(:, inliers));
% disp(A_refined);
% disp(t_refined);
%
% Author: Daniele Murgolo
% Date: March 1st, 2023
%
% See also: affine_test_case, estimate_affine, ransac_affine

[A, t] = estimate_affine(pts, pts_tilde);
end