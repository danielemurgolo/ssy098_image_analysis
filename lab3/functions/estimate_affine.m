function [A, t] = estimate_affine(pts, pts_tilde)
% Function Name: estimate_affine
%
% Description: This function estimates the affine transformation given a set of points and their
%              corresponding transformed points. The affine transformation is of the form
%              [x_tilde; y_tilde] = A*[x; y] + t.
%
% Inputs:
%   - pts: a 2xN matrix containing the original points
%   - pts_tilde: a 2xN matrix containing the transformed points
%
% Outputs:
%   - A: a 2x2 matrix representing the linear transformation of the affine transformation
%   - t: a 2x1 column vector representing the translation of the affine transformation
%
% Example Usage:
%   % generate some random points and apply a random affine transformation to them
%   >>[pts, pts_tilde, A_true, t_true] = affine_test_case();
%
%   % estimate the affine transformation from the original points to the transformed points
%   >>[A, t] = estimate_affine(pts, pts_tilde);
%
%   % apply the estimated affine transformation to the original points
%   >>pts_estimated = A*pts + t;
%
% Author: Daniele Murgolo
% Date: March 1st, 2023
%
% See also: affine_transform, RANSAC_algorithm

N = size(pts, 2);

%since we have that [x_tilde;y_tilde] = A*[x;y] + t we can write
%[x_tilde;y_tilde] = [a,b;c,d]*[x,y] + [t_x;t_y] which can be rewritten as
%[x_tilde;y_tilde] = [x,y,0,0,1,0; 0,0,x,y,0,1]*[a;b;c;d;t_x;t_y]
%we just have to stack all the points veritcally so the M matrix has size
%(2*N,6)
M = zeros(2*N, 6);
for j = 1:N

    M(2*j-1, 1:2) = pts(:, j);
    M(2*j-1, 5) = 1;

    M(2*j, 3:4) = pts(:, j);
    M(2*j, 6) = 1;

end

v = pts_tilde(:);

theta = M \ v;

A = [theta(1), theta(2); theta(3), theta(4)];
t = [theta(5); theta(6)];


end