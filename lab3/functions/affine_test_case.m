function [pts, pts_tilde, A_true, t_true] = affine_test_case()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
N = randi([3,200]);
A_true = rand(2,2);
t_true = randi(2,1)*100;
pts = rand(2, N);
pts_tilde = A_true*pts + t_true;
end