function U = minimal_triangulation(Ps, us)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
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