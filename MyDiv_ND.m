function y = MyDiv_ND(TV)

n_size_TV = size(TV);
n_size = n_size_TV(1 : end - 1);
n_size_TV_end = size(n_size, 2);

%y = zeros(n_size,'double','gpuArray');
y = zeros(n_size);

for k = 1 : size(n_size, 2)
    
    order_x = 1 : n_size_TV_end;
    order_x(1) = k;
    order_x(k) = 1;
    order_TV = [order_x n_size_TV_end+1];

    TV = reshape(permute(TV, order_TV), [n_size(k) prod(n_size)/n_size(k) n_size_TV_end]);
%     TV_permute=reshape(permute(TV,order_TV),[n_size(k) prod(n_size)/n_size(k) n_size_TV_end]);

    x_shift = circshift(TV(:, :, k), [1 0]);
    yx = TV(:, :, k) - x_shift;
    yx(1, :) = TV(1, :, k);
    yx(end, :) = -x_shift(end, :);

    yx = ipermute(reshape(yx, n_size(order_x)), order_x);
    TV = ipermute(reshape(TV, [n_size(order_x) n_size_TV_end]), order_TV);

    y = y + yx;
end

