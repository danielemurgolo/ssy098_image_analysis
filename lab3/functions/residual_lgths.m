function res = residual_lgths(A, t, pts, pts_tilde)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
pts_trans = A*pts + t;

M = pts_trans-pts_tilde;
res = sum(M.^2,1);
end