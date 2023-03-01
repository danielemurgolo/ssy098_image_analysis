function jacobian = compute_jacobian(Ps, U)
% Function Name: compute_jacobian
%
% Description: This function computes the Jacobian matrix for a 3D point U
%              and a set of camera projection matrices Ps. The Jacobian
%              matrix is used in bundle adjustment to iteratively refine
%              the 3D point estimate.
%
% Inputs:
%     - Ps: a cell array of length n representing the camera projection
%           matrices for n cameras
%     - U: a 3 x 1 vector representing the 3D point for which the Jacobian is
%          to be computed
%
% Outputs:
%     - jacobian: a 2N x 3 matrix representing the Jacobian matrix
%
% Example Usage:
%     >> Ps = {...} % define camera projection matrices
%     >> U = [...] % define 3D point
%     >> jacobian = compute_jacobian(Ps, U);
%     >> disp(jacobian);
%
% Author: Daniele Murgolo
% Date: March 1st, 2023
    N_cameras = length(Ps);
    
    jacobian = zeros(2*N_cameras, 3);

    U = [U;1];

    for i=1:2:2*N_cameras

        P = Ps{round(i/2)};
        a = P(1,:);
        b = P(2,:);
        c = P(3,:);

        j_i_x = (a(1:3)*(c*U) - (a*U)*c(1:3))/(c*U).^2;

        j_i_y = (b(1:3)*(c*U) - (b*U)*c(1:3))/(c*U).^2;

        jacobian(i,:) = j_i_x;
        jacobian(i+1,:) = j_i_y;

    end

end