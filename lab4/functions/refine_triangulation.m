function U = refine_triangulation(Ps, us, Uhat)
% Function Name: refine_triangulation
%
% Description: This function uses Gauss-Newton's method to refine an
%              approximate 3D point Uhat. It iteratively computes the
%              residuals and Jacobian matrices and updates the estimate
%              of Uhat until convergence.
%
% Inputs:
%     - Ps: a 3x4xM matrix representing the camera projection matrices
%           for M views
%     - us: a 2xN matrix representing the 2D image points corresponding
%           to the 3D point U in each of the M views
%     - Uhat: a 3x1 vector representing the approximate 3D point to be
%             refined
%
% Outputs:
%     - U: a 3x1 vector representing the refined 3D point
%
% Example Usage:
%     >> Ps = ... % define projection matrices
%     >> us = ... % define image points
%     >> Uhat = ... % define approximate 3D point
%     >> U = refine_triangulation(Ps, us, Uhat);
%     >> disp(U);
%
% Author: Daniele Murgolo
% Date: March 1st, 2023

    n_iter = 5;
    
    for i=1:n_iter

        r = compute_residuals(Ps,us,Uhat);
        J = compute_jacobian(Ps,Uhat);
        Uhat = Uhat - ((J'*J)\J')*r;

    end

    U = Uhat;

end