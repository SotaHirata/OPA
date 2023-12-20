function arrays = randarray_gen (N,num_randarray)
    arrays = zeros(N,N,num_randarray);

    for i = 1:num_randarray
        array = zeros(N);
        array(randperm(N^2, N))= 1;

        arrays(:,:,i) = array;
    end

end