function [x_k, varargout] = lad_reg_admm(C, b, x0, gamma, c, itmax, varargin)
%
% Solves the least absolute sum of abs problem using the ADMM algorithm
%
%  minimize ||C x - b ||_1 + gamma ||x||_1
%
%  in either the reals or complex domain depending on the type of C
%
% Input
%  C:     m x n matrix
%  b:     m x 1 vector
%  x0:    initialization n x 1 matrix
% gamma:  regularization
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

z_k = zeros(m + n, 1);
lambda_k = zeros(m + n, 1);

A = C'*C + gamma^2*eye(n);
V = chol(A);
%opts.UT = true;

alpha = 1.0;
if length(varargin) >= 1
    alpha = varargin{1};
end

fk = zeros(itmax, 1);
bp = [zeros(n, 1); b];
for k=1:itmax

    x_k = V\(V' \ ([gamma*eye(n), C']*(z_k - lambda_k + bp)));

    Cxk = C*x_k;
    
    
    if alpha == 1.0 
        r = Cxk - b;
        rp = [gamma*x_k; r];
        fk(k) = norm(rp, 1);
    else
        rp = alpha*([gamma*x_k;Cxk]) - (1 - alpha)*(-z_k - bp) - bp;
        fk(k) = norm(Cxk - b, 1) + gamma*norm(x_k, 1);
    end


    z_k = Soft(rp + lambda_k, 1/c);
    
    lambda_k = lambda_k + rp - z_k;

end

if nargout == 2
    info.fk = fk;
    varargout{1} = info;
end
