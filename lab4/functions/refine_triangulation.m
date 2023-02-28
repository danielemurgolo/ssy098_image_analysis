function U = refine_triangulation(Ps, us, Uhat)
%The function uses an approximate 3D point Uhat as a starting point
% for Gauss-Newton’s method

    n_iter = 5;
    
    for i=1:n_iter

        r = compute_residuals(Ps,us,Uhat);
        J = compute_jacobian(Ps,Uhat);
        Uhat = Uhat - ((J'*J)\J')*r;

    end

    U = Uhat;

end