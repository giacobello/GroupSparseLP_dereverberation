function y = Soft(x, t)
% l1-prox function = soft thresholding
mx = max(1- t./abs(x), 0);
y = x.*mx;
