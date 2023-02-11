function [pts, pts_tilde, A_true, t_true] = affine_test_case()
%The function generate N (between 3 and 200) points randomly
%and applies to them a random transformation
N = randi([3,200]);
A_true = rand(2,2);
t_true = randi(2,1)*100;
pts = rand(2, N);
pts_tilde = A_true*pts + t_true;
end