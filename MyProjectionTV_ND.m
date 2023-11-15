function p = MyProjectionTV_ND(g, tau, lam, iter, weights)

if (nargin < 5), weights = 1; end

n_size = size(g);
n_size_TV_end = size(n_size, 2);
pn = zeros([n_size n_size_TV_end],'double','gpuArray');
div_pn = zeros(n_size,'double','gpuArray');


for i = 1 : iter

    a = MyTV_ND_conv(div_pn - g ./ lam, weights);

    b = sqrt(sum(a .^ 2, n_size_TV_end + 1));
    pn = (pn + tau .* a) ./ (1.0 + tau .* b);
    
    div_pn = MyDiv_ND(pn);

end

p = lam .* MyDiv_ND(pn);

