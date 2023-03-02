function [pts, pts_tilde, A_true, t_true] = affine_test_case_outlier(outlier_rate)
% Function Name: affine_test_case_outlier
%
% Description: This function generates a test case for the affine transformation
% problem, by creating a set of random points and applying a random
% transformation to them. A specified percentage of the transformed points
% are replaced with outliers.
%
% Inputs:
%   - outlier_rate: A scalar between 0 and 1, indicating the proportion of
%       transformed points that should be replaced with outliers.
%
% Outputs:
%   - pts: A 2 x N matrix containing N random 2D points.
%   - pts_tilde: A 2 x N matrix containing the transformed points.
%   - A_true: A 2 x 2 matrix representing the true affine transformation
%       matrix.
%   - t_true: A 2 x 1 vector representing the true translation vector.
%
% Example Usage:
%   >> outlier_rate = 0.1;
%   >> [pts, pts_tilde, A_true, t_true] = affine_test_case_outlier(outlier_rate);
%
% Author: Daniele Murgolo
% Date: March 1st, 2023


N = randi([3, 200]);
n_outliers = round(N*outlier_rate);
A_true = randn(2, 2);
t_true = randn(2, 1) * 100;
pts = rand(2, N);
pts_tilde = A_true * pts + t_true;
outliers = N * randn(1, n_outliers);
idx_outliers = randperm(N, n_outliers);
pts_tilde(idx_outliers) = outliers;
end