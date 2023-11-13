function y = MyTVpsi_ND(x, th, tau, iter, N, weights, Regularizer, alpha)

if (nargin < 6), weights = 1; end
if (nargin < 7), Regularizer = 0; end
if (nargin < 8), alpha = 0; end

X = reshape(x, N);

Y = X - MyProjectionTV_ND(X, tau, th*0.5, iter, weights);

% y = Y;
y = reshape(Y, prod(N), 1);

if isa(Regularizer, 'function_handle')
    y = (Regularizer(y) + alpha * y(:)) ./ (1 + alpha);
    y = y(:);
end
