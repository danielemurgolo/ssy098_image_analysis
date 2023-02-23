function [U, nbr_inliers] = ransac_triangulation(Ps, us, threshold)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
N_cameras = length(Ps);
U = zeros(3,N_cameras);
prob = 0.995;
max_trials = 100;
hard_limit = 2e5;
n_trials = 0;
n_inliers_best = 0;
best_residuals = 0;
nbr_inliers = 0;
n_samples = 2;

while n_trials < max_trials && n_trials<hard_limit

    n_trials = n_trials+1;
    idx = randperm(N_cameras, n_samples);

    sample_Ps = Ps(idx);
    sample_us = us(:,idx);

    U_min = minimal_triangulation(sample_Ps, sample_us);

    res = reprojection_errors(Ps, us, U_min);

    n_inliers = sum(res<=threshold);

    if n_inliers>n_inliers_best

        n_inliers_best = n_inliers;
        U = U_min;
        nbr_inliers = n_inliers_best;

        eps = n_inliers/N_cameras;

        max_trials = abs(int32(log(1 - prob) / log(1 - eps.^n_samples)));

        best_residuals = mean(res);

    end

end
end