function jacobian = compute_jacobian(Ps, U)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    N_cameras = length(Ps);
    
    jacobian = zeros(2*N_cameras, 3);

    U = [U;1];

    for i=1:2:N_cameras

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