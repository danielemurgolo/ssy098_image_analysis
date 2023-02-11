function res = residual_lgths(A, t, pts, pts_tilde)
%The function computes the squared length of the 2D residuals
pts_trans = A*pts + t;

M = pts_trans-pts_tilde;
res = sum(M.^2,1);
end