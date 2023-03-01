function U = minimal_triangulation(Ps, us)
% Function Name: minimal_triangulation
%
% Description: 
%   This function takes in two camera matrices, Ps, and two image points, 
%   us, and triangulates a 3D point.
%
% Inputs:
%   Ps - cell array of two 3 x 4 camera matrices
%   us - 2 x 2 matrix of image points (us(:,1) corresponds to Ps{1}, and
%        us(:,2) corresponds to Ps{2})
%
% Output:
%   U - 3x1 column vector representing the 3D point
%
% Example Usage:
%     >> Ps = {...} % define camera projection matrices
%     >> us = [...] % define image points
%     >> U = minimal_triangulation(Ps, us)
%     >> disp(U);
%
% Author: Daniele Murgolo
% Date: March 1st, 2023
    x1 = us(1, 1);
    y1 = us(2, 1);
    x2 = us(1, 2);
    y2 = us(2, 2);
    Pl = Ps{1};
    P2 = Ps{2};
    % The equations: us{1} = Ps{1} * X and us{2} = Ps{2} * X, can be
    % combined into the matrix A.
    A = [x1 * Pl(3,:) - Pl(1,:);
         y1 * Pl(3,:) - Pl(2,:);
         x2 * P2(3,:) - P2(1,:);
         y2 * P2(3,:) - P2(2,:)];
    % We now need to solve the homogeneous equation: Ax = 0, so x can be
    % found as the last column of the V component of the SVD of A.
    [U, S, V] = svd(A);
    % Finally, the last component needs to become 1, so we normalize.
    U = V(1:3, end) / V(end, end);
end