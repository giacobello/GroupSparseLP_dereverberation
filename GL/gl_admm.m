function [Xk, varargout] = gl_admm(C, B, Z0, c, itmax, varargin)
%
% Solves the group sparsity problem using the ADMM algorithm
%
%  minimize ||C X - B ||_2,1 = \sum_i=1^m ||C_i X - B_i||_2
%
%  in either the reals or complex domain depending on the type of C
%
% Input
%  C:     m x n matrix
%  Z:     m x k vector
%  Z0:    initialization n x k matrix
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

Zk = Z0;
lambdak = zeros(size(B));

[Q, RR] = qr(C, 0);
opts.UT = true;

alpha = 1.0;
if length(varargin) >= 1
    alpha = varargin{1};
end

fk = zeros(itmax, 1);

for k=1:itmax
    %Xk = linsolve(RR, Q'*(Zk + B - lambdak), opts);
    Xk = RR \ (Q'*(Zk + B - lambdak));

    CXk = C*Xk;
    if alpha == 1.0 
        R = CXk - B;
        fk(k) = sum(sqrt(sum(abs(R).^2, 2)));
    else
        R = alpha*(CXk) - (1 - alpha)*(-Zk - B) - B;
        fk(k) = sum(sqrt(sum(abs(CXk - B).^2, 2)));
    end

    Zk = SV(R + lambdak, 1/c);

    lambdak = lambdak + R - Zk;
end

if nargout == 2
    info.fk = fk;
    info.Zk = Zk;
    varargout{1} = info;
end
