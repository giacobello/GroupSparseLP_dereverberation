function tests = test()
%
% Test script for evaluating the LAD dereverberation formulation
%
%  Can be executed as runtests('test')
%
% Author:
%    T. L. Jensen,  tlj@es.aau.dk    
%    Aalborg University, 2017
%      

% Seeding may be used to make debugging easier.
    rng(100);

    tests = functiontests(localfunctions);
end    

function test_S_real(testCase)
% Testing of gl softhresholding
    n = 10;
    x = randn(n, 1);
    t = 1.1;

    cvx_begin
    cvx_quiet(true)
    variable u(n)
    minimize t*norm(u, 2) + 0.5*square_pos(norm(u - x, 2))
    cvx_end

    uh = S(x, t);

    testCase.assertEqual(u, uh, 'absTol', 1e-4)
end

function test_S_complex(testCase)
% Testing of gl softhresholding
    n = 10;
    x = randn(n, 1) + 1j*randn(n, 1);
    t = 1.1;

    cvx_begin
    cvx_quiet(true)
    variable u(n) complex
    minimize t*norm(u, 2) + 0.5*square_pos(norm(u - x, 2))
    cvx_end

    uh = S(x, t);

    testCase.assertEqual(u, uh, 'absTol', 1e-4)
end
%
function test_SV_complex(testCase)
% Testing of vectorized softhresholding
    k = 10;
    n = 100;
    x = randn(n, k) + 1j*randn(n, k);
    t = 1.1;

    u = 0*x;
    for nn = 1:n
        u(nn, :) = S(x(nn, :), t);
    end
    
    uh = SV(x, t);

    testCase.assertEqual(u, uh, 'absTol', 1e-4)
end

function test_random_real(testCase)

    m = 100;
    n = 30;
    k = 4;
    
    C = randn(m, n);
    B = randn(m, k);
    
    cvx_begin
    cvx_quiet(true)
    variable X(n, k)
    minimize sum(norms(C*X - B, 2, 2))
    cvx_end
   
    fs = sum(sqrt(sum(abs(C*X - B).^2, 2)));
    
    testCase.assertEqual(fs, sum(norms(C*X - B, 2, 2)), 'absTol', 1e-3)
    
    Xp = gl_admm(C, B, zeros(size(B)), 1.7, 1000);
    
    testCase.assertEqual(X, Xp, 'absTol', 1e-3)
end


function test_random_complex(testCase)

    m = 100;
    n = 30;
    k = 4;
    
    C = randn(m, n) + 1j*randn(m, n);
    B = randn(m, k) + 1j*randn(m, k);
    
    cvx_begin
    cvx_quiet(true)
    variable X(n, k) complex
    minimize sum(norms(C*X - B, 2, 2))
    cvx_end

    fs = sum(sqrt(sum(abs(C*X - B).^2, 2)));
    
    testCase.assertEqual(fs, sum(norms(C*X - B, 2, 2)), 'absTol', 1e-3)
    
    Xp = gl_admm(C, B, zeros(size(B)), 1.7, 1000);
    
    testCase.assertEqual(X, Xp, 'absTol', 1e-3)

end
%
function test_dereverb_complex_case(testCase)

    load('DeRev_Example.mat')

    M = size(Xref, 2);
    LgM = size(XX, 2);
    
    cvx_begin quiet
    variable Gk(LgM, M) complex
    minimize(  sum(norms(XX*Gk - Xref, 2, 2)) )
    cvx_end
    
    [Gkp, info] = gl_admm(XX, Xref, zeros(size(Xref)), 1.7, 1000);
    
    f = @(G) sum(sqrt(sum(abs(XX*G - Xref).^2, 2)));
    
    testCase.assertEqual(f(Gk), f(Gkp), 'relTol', 1e-4)
    testCase.assertEqual(f(Gk), info.fk(end), 'relTol', 1e-5)
    testCase.assertEqual(Gk, Gkp, 'relTol', 1e-3)

end


function test_random_reg_real(testCase)

    m = 100;
    n = 30;
    k = 4;
    
    C = randn(m, n);
    B = randn(m, k);
    
    alpha = 2.2;
    
    cvx_begin
    cvx_quiet(true)
    variable X(n, k)
    minimize sum(norms(C*X - B, 2, 2)) + ...
        alpha * sum(sum(abs(X)))
    cvx_end
   
    f = @(X) sum(sqrt(sum(abs(C*X - B).^2, 2))) + alpha * sum(sum(abs(X)));
    
    testCase.assertEqual(f(X), sum(norms(C*X - B, 2, 2)) + alpha * sum(sum(abs(X))), 'absTol', 1e-6)
    
    [Xp, info] = gl_reg_admm(C, B, zeros(size(B)), alpha, 1.7, 1000);

    testCase.assertEqual(f(X), f(Xp), 'relTol', 1e-5)
    testCase.assertEqual(f(X), info.fk(end), 'relTol', 1e-5)
    testCase.assertEqual(X, Xp, 'absTol', 1e-3)
    
end


function test_random_reg_complex(testCase)

    m = 100;
    n = 30;
    k = 4;
    
    C = randn(m, n) + 1j*randn(m, n);
    B = randn(m, k) + 1j*randn(m, k);
    
    alpha = 1.5;
    cvx_begin
    cvx_quiet(true)
    variable X(n, k) complex
    minimize sum(norms(C*X - B, 2, 2)) + ...
        alpha * sum(sum(abs(X)))
    cvx_end

    f = @(X) sum(sqrt(sum(abs(C*X - B).^2, 2))) + alpha * sum(sum(abs(X)));

    testCase.assertEqual(f(X), sum(norms(C*X - B, 2, 2)) + alpha * sum(sum(abs(X))), 'absTol', 1e-6)
    
    [Xp, info] = gl_reg_admm(C, B, zeros(size(B)), alpha, 2, 1000, 1.7);

    testCase.assertEqual(f(X), f(Xp), 'relTol', 1e-5)
    testCase.assertEqual(f(X), info.fk(end), 'relTol', 1e-5)
    testCase.assertEqual(X, Xp, 'absTol', 1e-3)

end
%
function test_dereverb_complex_reg_case(testCase)

    load('DeRev_Example.mat')

    M = size(Xref, 2);
    LgM = size(XX, 2);
    
    alpha = 3.2;
    
    cvx_begin quiet
    variable Gk(LgM, M) complex
    minimize(  sum(norms(XX*Gk - Xref, 2, 2)) + ...
                       alpha * sum(sum(abs(Gk))) )
    cvx_end
    
    [Gkp, info] = gl_reg_admm(XX, Xref, zeros(size(Xref)), alpha, 2, 1000, 1.7);
    
    f = @(X) sum(sqrt(sum(abs(XX*Gk - Xref).^2, 2))) + alpha * sum(sum(abs(Gk)));
    
    testCase.assertEqual(f(Gk), f(Gkp), 'relTol', 1e-4)
    testCase.assertEqual(f(Gk), info.fk(end), 'relTol', 1e-5)
    testCase.assertEqual(Gk, Gkp, 'absTol', 1e-3)

end
