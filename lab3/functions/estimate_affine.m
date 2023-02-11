function [A, t] = estimate_affine(pts, pts_tilde)
%The function estimates the affine trasformation given
%some points and the respective transformed points
N = size(pts,2);

%since we have that [x_tilde;y_tilde] = A*[x;y] + t we can write
%[x_tilde;y_tilde] = [a,b;c,d]*[x,y] + [t_x;t_y] which can be rewritten as
%[x_tilde;y_tilde] = [x,y,0,0,1,0; 0,0,x,y,0,1]*[a;b;c;d;t_x;t_y]
%we just have to stack all the points veritcally so the M matrix has size
%(2*N,6)
M = zeros(2*N,6);
for j=1:N

    M(2*j-1,1:2) = pts(:,j);
    M(2*j-1,5) = 1;

    M(2*j,3:4) = pts(:,j);
    M(2*j,6) = 1;

end

v = pts_tilde(:);

theta = M\v;

A = [theta(1), theta(2); theta(3),theta(4)];
t = [theta(5);theta(6)];
    


end