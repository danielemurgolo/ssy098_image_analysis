function [A, t] = ransac_fit_affine(pts, pts_tilde, threshold)
% Function Name: ransac_fit_affine
%
% Description: This function estimates the affine transformation that maps
%              a set of 2D points `pts` to their corresponding set of 2D points
%              `pts_tilde` using the RANSAC algorithm. The algorithm is used
%              to remove the effect of outliers in the input data. The function
%              works by randomly selecting subsets of 3 input point pairs,
%              computing the affine transformation using the selected pairs,
%              and then computing the number of inliers (point pairs that
%              lie within a distance threshold of the estimated transformation).
%              The best transformation that maximizes the number of inliers
%              is returned.
%
% Inputs:
%   - pts: a 2xN matrix representing the input set of 2D points
%   - pts_tilde: a 2xN matrix representing the transformed set of 2D points
%   - threshold: the distance threshold used to determine if a point pair is
%     an inlier or outlier
%
% Outputs:
%   - A: a 2x2 matrix representing the estimated affine transformation
%   - t: a 2x1 vector representing the estimated translation vector
%
% Example Usage:
%   % Generate test case
%   [pts, pts_tilde, A_true, t_true] = affine_test_case_outlier(0.2);
%
%   % Estimate affine transformation using RANSAC
%   [A, t] = ransac_fit_affine(pts, pts_tilde, 0.1);
%
%   % Compute residual errors
%   res = residual_lgths(A, t, pts, pts_tilde);
%
%   % Plot results
%   figure;
%   plot(pts(1,:), pts(2,:), 'ro');
%   hold on;
%   plot(pts_tilde(1,:), pts_tilde(2,:), 'bo');
%   plot(pts_tilde(1,:), pts_tilde(2,:), 'b+');
%   plot(pts_trans(1,:), pts_trans(2,:), 'g+');
%   title('RANSAC-based affine transformation');
%   legend('Input points', 'Transformed points', 'Outliers', 'Inliers');
%
% References:
%   - Fischler, M. A., & Bolles, R. C. (1981). Random sample consensus: a
%     paradigm for model fitting with applications to image analysis and
%     automated cartography. Communications of the ACM, 24(6), 381-395.
%
% Author: Daniele Murgolo
% Date: March 1st, 2023

A = zeros(2, 2);
t = zeros(2, 1);

n_samples = 3;
N = size(pts, 2);
p = 0.99;
hard_max = 1e5;
max_iter = hard_max;
i = 0;
n_inliers_best = 0;
best_res = 0;
while i < max_iter && i < N

    i = i + 1;
    idx = randperm(N, n_samples);
    sample_pts = pts(:, idx);
    sample_pts_tilde = pts_tilde(:, idx);

    [A_sample, t_sample] = estimate_affine(sample_pts, sample_pts_tilde);

    res = residual_lgths(A_sample, t_sample, sample_pts, sample_pts_tilde);

    n_inliers = sum(res < threshold);

    if n_inliers > n_inliers_best

        n_inliers_best = n_inliers;
        eps = n_inliers / N;
        max_iter = abs(int32(log(1-p)/log(1-eps.^n_samples)));
        A = A_sample;
        t = t_sample;
        best_res = res;

    end

end

end