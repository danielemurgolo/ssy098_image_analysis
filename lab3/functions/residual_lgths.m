function res = residual_lgths(A, t, pts, pts_tilde)
% Function Name: residual_lgths
%
% Description: This function takes an estimated affine transformation A and a translation vector t, along with a set of
%              input points pts and their transformed counterparts pts_tilde. The function then computes the squared length
%              of the 2D residuals between the transformed points and the estimated points, and returns the result as res.
%
% Inputs:
%   - A: The estimated 2x2 affine transformation matrix
%   - t: The estimated 2x1 translation vector
%   - pts: A 2xN matrix containing N input points
%   - pts_tilde: A 2xN matrix containing N transformed points corresponding to the input points
%
% Outputs:
%   - res: A 1xN row vector containing the squared length of the 2D residuals for each point in the input set
%
% Example Usage:
%   >> [pts, pts_tilde, A_true, t_true] = affine_test_case();
%   >> [A_est, t_est] = estimate_affine(pts, pts_tilde);
%   >> residuals = residual_lgths(A_est, t_est, pts, pts_tilde);
%   >> disp(residuals);
%
% Author: Daniele Murgolo
% Date: March 1st, 2023
%
% See also: affine_test_case, estimate_affine

pts_trans = A * pts + t;

M = pts_trans - pts_tilde;
res = sum(M.^2, 1);
end