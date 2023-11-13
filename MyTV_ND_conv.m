function TV = MyTV_ND_conv(x, weights)

if (nargin < 2), weights = 1; end

n_size = size(x);
n_size_TV_end = size(n_size, 2);
TV = zeros([n_size n_size_TV_end]);


for k = 1 : n_size_TV_end
    
    order_x = 1 : n_size_TV_end;
    order_x(1) = k;
    order_x(k) = 1;
    order_TV = [order_x n_size_TV_end+1];

    x = reshape(permute(x, order_x), [n_size(k) prod(n_size)/n_size(k)]);
%     x_permute=reshape(permute(x,order_x),[n_size(k) prod(n_size)/n_size(k)]);
    TV = reshape(permute(TV, order_TV), [n_size(k) prod(n_size)/n_size(k) n_size_TV_end]);
    
    TV(:, :, k) = circshift(x, [-1 0]) - x;
    TV(end, :, k) = 0.0;

    x = ipermute(reshape(x, n_size(order_x)), order_x);
    TV = ipermute(reshape(TV, [n_size(order_x) n_size_TV_end]), order_TV);
    
end


n_size_weights = size(weights);

if((n_size_weights(1) == 1) && (n_size_weights(2) == n_size_TV_end))
    for k = 1 : n_size_TV_end
        
        order_x = 1 : n_size_TV_end;
        order_x(1) = k;
        order_x(k) = 1;
        order_TV = [order_x n_size_TV_end+1];
        
        TV = reshape(permute(TV,order_TV), [n_size(k) prod(n_size)/n_size(k) n_size_TV_end]);        
        TV(:, :, k) = TV(:, :, k) .* weights(k);
        
        TV = ipermute(reshape(TV, [n_size(order_x) n_size_TV_end]), order_TV);

    end
else
    TV = TV .* weights;
end
