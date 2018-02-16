function [x_k, varargout] = lad_admm(C, b, x0, c, itmax, varargin)
%
% Solves the least absolute sum of abs problem using the ADMM algorithm
%
%  minimize ||C x - b ||_1
%
%  in either the reals or complex domain depending on the type of C
%
% Input
%  C:     m x n matrix
%  b:     m x 1 vector
%  x0:    initialization n x 1 matrix
%  c:     penalty parameter
%  itmax  maximum number of iterations
%  alpha: (default 1.0) over/under relaxation alpha \in (0, 2)
%  
% Output:
%  x_k:   solution after k=itmax iterations
%  info:  if requested, will contain
%          info.fk = the objective for each iteration
%
% Version:

[m, n] = size(C);

z_k = zeros(m, 1);
lambda_k = zeros(m, 1);

[Q, R] = qr(C, 0);
opts.UT = true;

alpha = 1.0;
if length(varargin) >= 1
    alpha = varargin{1};
end

fk = zeros(itmax, 1);

for k=1:itmax
    %x_k = linsolve(R, Q'*(z_k + b - lambda_k), opts);
    x_k = R \ (Q'*(z_k + b - lambda_k));

    Cxk = C*x_k;
    if alpha == 1.0 
        r = Cxk - b;
        fk(k) = norm(r, 1);
    else
        r = alpha*(Cxk) - (1 - alpha)*(-z_k - b) - b;
        fk(k) = norm(Cxk - b, 1);
    end

    z_k = Soft(r + lambda_k, 1/c);

    lambda_k = lambda_k + r - z_k;

end

if nargout == 2
    info.fk = fk;
    varargout{1} = info;
end
