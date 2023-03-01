function positive = check_depths(Ps, U)
% Function Name: check_depths
%
% Description: This function takes a set of camera projection matrices and
%              a 3D point as input, and checks whether the point has positive
%              depth in each of the cameras. The output is an array of boolean
%              values indicating which depths were positive.
%
% Inputs:
%     - Ps: a cell array of length n representing the camera projection
%           matrices for n cameras
%     - U: a 3 x 1 vector representing the 3D point to be checked
%
% Outputs:
%     - positive: a 1 x N boolean array representing whether the 3D point has
%                 positive depth in each of the n cameras
%
% Example Usage:
%     >> Ps = {...} % define camera projection matrices
%     >> U = [...] % define 3D point
%     >> positive = check_depths(Ps, U);
%     >> disp(positive);
%
% Author: Daniele Murgolo
% Date: March 1st, 2023

n_cameras = length(Ps);
positive = [];
for i = 1:n_cameras


    x = Ps{i} * [U; 1];

    if x(3) < 0
        positive = [positive, false];

    else
        positive = [positive, true];

    end
end

end