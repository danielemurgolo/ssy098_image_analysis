function [A,t] = ransac_fit_affine(pts, pts_tilde, threshold)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
A = zeros(2,2);
t = zeros(2,1);

n_samples = 3;
N = size(pts,2);
p = 0.99;
hard_max = 1e5;
max_iter = hard_max;
i = 0;
n_inliers_best = 0;
best_res = 0;
while i<max_iter && i<N

    i = i+1;
    idx = randperm(N,n_samples);
    sample_pts = pts(:,idx);
    sample_pts_tilde = pts_tilde(:,idx);

    [A_sample, t_sample] = estimate_affine(sample_pts,sample_pts_tilde);

    res = residual_lgths(A_sample, t_sample, sample_pts, sample_pts_tilde);

    n_inliers = sum(res<threshold);

    if n_inliers>n_inliers_best

        n_inliers_best = n_inliers;
        eps = n_inliers/N;
        max_iter = abs(int32(log(1 - p) / log(1 - eps.^n_samples)));
        A = A_sample;
        t = t_sample;
        best_res = res;

    end

end

end