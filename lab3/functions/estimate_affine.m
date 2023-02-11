function [A, t] = estimate_affine(pts, pts_tilde)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
N = size(pts,2);

M = zeros(2*N,6);

for j=1:N

    M(2*j-1,1:2) = pts(:,j);
    M(2*j-1,5) = 1;

    M(2*j,3:4) = pts(:,j);
    M(2*j,6) = 1;

end

v = pts_tilde(:);

theta = M\v;

A = [theta(1), theta(2); theta(3),theta(4)];
t = [theta(5);theta(6)];
    


end