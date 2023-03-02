function [pts, pts_tilde, A_true, t_true] = affine_test_case()
% Function Name: affine_test_case
%
% Description: This function generates N points randomly, where N is a random
%              integer between 3 and 200, and applies to them a random affine transformation.
%              The function returns the original points, transformed points, and the true
%              affine transformation that was applied.
%
% Outputs:
%   - pts: a 2xN matrix containing the randomly generated points
%   - pts_tilde: a 2xN matrix containing the transformed points
%   - A_true: a 2x2 matrix representing the true linear transformation applied to the points
%   - t_true: a 2x1 column vector representing the true translation applied to the points
%
% Example Usage:
%   % generate some random points and apply a random affine transformation to them
%   >>[pts, pts_tilde, A_true, t_true] = affine_test_case();
%
% Author: Daniele Murgolo
% Date: March 1st, 2023
%
% See also: affine_transform, RANSAC_algorithm

N = randi([3, 200]);
A_true = randn(2, 2);
t_true = randn(2, 1) * 100;
pts = rand(2, N);
pts_tilde = A_true * pts + t_true;
end