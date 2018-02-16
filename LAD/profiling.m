
load('DeRev_LSA_Example.mat')

M = size(Xref, 2);
LgM = size(XX, 2);
    
Gkp = zeros(LgM, M);


profile on
tic;
for m = 1:M
    Gkp(:, m) = lad_admm(XX, Xref(:, m), zeros(size(XX, 1), 1), 1.7, 1000);
end
toc
profile off


