function all_residuals = compute_residuals(Ps, us, U)
% Function Name: compute_residuals
%
% Description: This function computes the reprojection residuals for a 3D
%              point U in N camera projections, given the camera projection
%              matrices Ps and the corresponding image points us.
%
% Inputs:
%     - Ps: a cell array of length N representing the camera projection
%           matrices for N cameras
%     - us: a 2 x N array representing the image points for each camera
%     - U: a 3 x 1 vector representing the 3D point to be projected
%
% Outputs:
%     - all_residuals: a 2N x 1 array of all the reprojection residuals
%
% Example Usage:
%     >> Ps = {...} % define camera projection matrices
%     >> us = [...] % define image points
%     >> U = [...] % define 3D point
%     >> residuals = compute_residuals(Ps, us, U);
%     >> disp(residuals);
%
% Author: Daniele Murgolo
% Date: March 1st, 2023
N_cameras = length(Ps);
Us = [U; 1];
all_residuals = zeros(2*N_cameras, 1);

for i = 1:2:2 * N_cameras

    P = Ps{round(i/2)};
    a = P(1, :);
    b = P(2, :);
    c = P(3, :);
    positive = check_depths(Ps, U);

    if positive

        x = (a * Us) / (c * Us) - us(1, round(i/2));
        y = (b * Us) / (c * Us) - us(2, round(i/2));
        all_residuals(i) = x;
        all_residuals(i+1) = y;

    else

        all_residuals(i) = inf;
        all_residuals(i+1) = inf;

    end

end

end