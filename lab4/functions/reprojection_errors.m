function errors = reprojection_errors(Ps, us, U)
% Function Name: check_depths
%
% Description: This function first checks if each 3D point has positive 
%              depth (i.e., is in front of all the cameras), and sets the
%              reprojection error to infinity if not. For points with positive
%              depth, the function computes the reprojection error as the 
%              Euclidean distance between the 2D projection of the 3D point
%              and the corresponding image point. The output is a vector of
%              reprojection errors, with one entry for each camera.
%
% Inputs:
%   - Ps: A cell array of N camera matrices, where each matrix is 3 x 4.
%   - us: A 2 x N matrix of N 2D image points.
%   - U: A 3 x M matrix of M 3D points.
%
% Outputs:
%   - errors: A N x 1 vector of reprojection errors, where N is the number of
%             cameras. If a point has negative depth, its reprojection error
%             is set to Inf.
%
% Example Usage:
%     >> Ps = {...} % define camera projection matrices
%     >> U = [...] % define 3D point
%     >> us = [...]; % 2 x N
%     >> errors = reprojection_errors(Ps, us, U)
%     >> disp(errors);
%
% Author: Daniele Murgolo
% Date: March 1st, 2023
    N_cameras = length(Ps);
    positive = check_depths(Ps, U);
    errors = zeros(N_cameras,1);
    
    for i=1:N_cameras
    
        if positive(i)

            u = Ps{i}*[U;1];
            u = u(1:2)/u(3);
            errors(i) = sqrt(sum(us(:,i)-u).^2);

        else
            errors(i) = inf;

        end
    
    end
end