function [st, m, v] = Adam_func(st, m, v, t, alpha, beta_1, beta_2, epsilon)
%入力
%st:損失関数の勾配
%m:前ステップのm（勾配の移動平均）→初期値0（次元はstと同じ）
%v:前ステップのm（勾配の2乗の移動平均）→初期値0（次元はstと同じ）
%t:今何ステップ目か→初期値1
%alpha:最急降下法とかで使ってた値
%beta_1:mに関する学習係数（0〜1)→初期値0.9
%beta_2:mに関する学習係数（0〜1)→初期値0.999
%epsilon:1e-8とか？

alpha_t = alpha * sqrt(1 - beta_2^t) / (1 - beta_1^t);

m = beta_1 * m + (1 - beta_1) * st;
v = beta_2 * v + (1 - beta_2) * (real(st) .^ 2 + 1i * imag(st) .^ 2);
% v = beta_2 * v + (1 - beta_2) * (abs(st) .^ 2);

st = alpha_t * (real(m) ./ (sqrt(real(v)) + epsilon) + 1i * imag(m) ./ (sqrt(imag(v)) + epsilon));
% st = alpha_t * (m ./ (sqrt(v) + epsilon));

% m = mean(m, 3);
% v = mean(v, 3);

end