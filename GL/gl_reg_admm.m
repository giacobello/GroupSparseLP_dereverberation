function [Xk, varargout] = gl_reg_admm(C, B, Z0, gamma, c, itmax, varargin)
%
% Solves the group sparsity problem using the ADMM algorithm
%
%  minimize ||C X - B ||_2,1 + alpha ||X||_1,1
%        = \sum_i=1^m ||C_i X - B_i||_2 + \sum_i=1^m \sum_j=1^k |X_ij|
%
%  in either the reals or complex domain depending on the type of C
%
% Input
%  C:     m x n matrix
%  Z:     m x k vector
%  Z0:    initialization n x k matrix
% gamma:  regularization parameter  
%  c:     penalty parameter
%  itmax  maximum number of iterations
%  alpha: (default 1.0) over/under relaxation alpha \in (0, 2)
%  
% Output:
%  Xk:   solution after k=itmax iterations
%  info:  if requested, will contain
%          info.fk = the objective for each iteration
%          info.Zk = the dual variable at k=itmax iterations


[m, n] = size(C);
k = size(B, 2);
Bp = [zeros(n, k); B];

Zk = zeros(size(Bp));
lambdak = zeros(size(Bp));

A = C'*C + gamma^2*eye(n);
V = chol(A);

[Q, RR] = qr(C, 0);

alpha = 1.0;
if length(varargin) >= 1
    alpha = varargin{1};
end

fk = zeros(itmax, 1);

for k=1:itmax
    
    Xk = V\(V' \ ([gamma*eye(n), C']*(Zk + Bp - lambdak)));

    CXk = C*Xk;
    if alpha == 1.0 
        R = CXk - B;
        Rp = [gamma*Xk; R];
        fk(k) = sum(sqrt(sum(abs(R).^2, 2))) + gamma*sum(sum(abs(Xk)));
    else
        Rp = alpha*([gamma*Xk;CXk]) - (1 - alpha)*(-Zk - Bp) - Bp;
        fk(k) = sum(sqrt(sum(abs(CXk - B).^2, 2))) +  gamma*sum(sum(abs(Xk)));

        %R = alpha*(CXk) - (1 - alpha)*(-Zk - B) - B;
        %fk(k) = sum(sqrt(sum(abs().^2, 2)));
    end

    Zk = [Soft(Rp(1:n,:)+lambdak(1:n,:), 1/c); SV(Rp(n+1:end, :) + lambdak(n+1:end,:), 1/c)];

    lambdak = lambdak + Rp - Zk;
end

if nargout == 2
    info.fk = fk;
    info.Zk = Zk;
    varargout{1} = info;
end
