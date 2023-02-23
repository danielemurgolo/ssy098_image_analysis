function U = minimal_triangulation(Ps, us)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
x1 = us(1,1);
y1 = us(2,1);
x2 = us(2,1);
y2 = us(2,2);

P1 = Ps{1};
P2 = Ps{2};

A = [x1 * P1(3,:) - P1(1,:);
     y1 * P1(3,:) - P1(2,:);
     x2 * P2(3,:) - P2(1,:);
     y2 * P2(3,:) - P2(2,:)];

[U,S,V] = svd(A);

U = V(1:3, end)/V(end,end);
end