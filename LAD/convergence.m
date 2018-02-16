

load('DeRev_LSA_Example.mat')

M = size(Xref, 2);
LgM = size(XX, 2);

itmax = 200;

m = 2;
[Gkp, info0] = lad_admm(XX, Xref(:, m), zeros(size(XX, 1), 1), 100, 10*itmax);

[Gkp, info1] = lad_admm(XX, Xref(:, m), zeros(size(XX, 1), 1), 50, itmax);

[Gkp, info2] = lad_admm(XX, Xref(:, m), zeros(size(XX, 1), 1), 50, ...
                        itmax, 1.7);

fs = info0.fk(end);

figure(1)
clf
semilogy((info1.fk - fs)/fs)
hold on;
plot((info2.fk - fs)/fs)
legend('alpha=1.0', 'overrelaxation')