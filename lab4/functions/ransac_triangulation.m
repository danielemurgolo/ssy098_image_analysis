function [U, nbr_inliers] = ransac_triangulation(Ps, us, threshold)
% Function Name: ransac_triangulation
%
% Description:
%   This function performs RANSAC-based triangulation to estimate the 3D
%   location of a point given multiple camera matrices and their corresponding
%   2D image points. It randomly selects a minimum number of camera matrices
%   and their corresponding image points, and calculates the 3D point using
%   the 'minimal_triangulation' function. It then computes the reprojection
%   errors for all the camera matrices and 2D image points. The 3D point with
%   the highest number of inliers (reprojection errors within a threshold) is
%   selected as the final estimate. The function returns the estimated 3D
%   point and the number of inliers found.
%
% Inputs:
%   Ps - A cell array of N cameras 3 x 4 camera matrices
%   us - A 2 x N matrix of N cameras image points (each column corresponds to a
%        camera matrix in Ps)
%   threshold - A scalar threshold to determine if a reprojection error is an
%               outlier or an inlier.
%
% Outputs:
%   U - A 3 x 1 column vector representing the estimated 3D point
%   nbr_inliers - The number of inliers (reprojection errors within threshold)
%                 found for the estimated 3D point.
%
% Example Usage:
%     >> Ps = {...}; %
%     >> us = [...]; % 2 x N
%     >> threshold = 1.5;
%     >> [U, nbr_inliers] = ransac_triangulation(Ps, us, threshold);
%
% Author: Daniele Murgolo
%
% Date: March 1st, 2023

N_cameras = length(Ps);
U = zeros(3, N_cameras);
prob = 0.995;
max_trials = 100;
hard_limit = 2e5;
n_trials = 0;
n_inliers_best = 0;
best_residuals = 0;
nbr_inliers = 0;
n_samples = 2;

while n_trials < max_trials && n_trials < hard_limit

    n_trials = n_trials + 1;
    idx = randperm(N_cameras, n_samples);

    sample_Ps = Ps(idx);
    sample_us = us(:, idx);

    U_min = minimal_triangulation(sample_Ps, sample_us);

    res = reprojection_errors(Ps, us, U_min);

    n_inliers = sum(res <= threshold);

    if n_inliers > n_inliers_best

        n_inliers_best = n_inliers;
        U = U_min;
        nbr_inliers = n_inliers_best;

        eps = n_inliers / N_cameras;

        max_trials = abs(int32(log(1-prob)/log(1-eps.^n_samples)));

        best_residuals = mean(res);

    end

end
end