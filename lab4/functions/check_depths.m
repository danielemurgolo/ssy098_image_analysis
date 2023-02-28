function positive = check_depths(Ps, U)
% The function takes n_camera matrices, Ps, and a 3D point, U,
% and checks the depth of U in each of the cameras. The output is
% an array of boolean values of length n_camera that indicates which depths
% were positive
    n_cameras = length(Ps);
    positive = [];
    for i=1:n_cameras
    
    
            x = Ps{i}*[U;1];
    
            if x(3) < 0
                positive = [positive, false];
    
            else
                positive = [positive, true];
    
            end
    end

end