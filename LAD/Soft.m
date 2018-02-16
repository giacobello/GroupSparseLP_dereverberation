function y = Soft(x, t)
% l1-prox function = soft thresholding
mx = max(abs(x) - t, 0);
y = sign(x).*mx;
