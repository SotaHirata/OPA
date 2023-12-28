function matrix = Gaussianarray_gen(N, M, sigma)
    % N×N行列を0で初期化
    matrix = zeros(N, N);
    
    % 行列の中心座標
    center = ceil(N / 2);
    
    % ガウス分布に従ってサンプリングされたM点に1を代入
    for i = 1:M
        success = false;
        attempts = 0;
        
        while ~success
            % ガウス分布に従って座標をサンプリング
            x = round(normrnd(center, sigma));
            y = round(normrnd(center, sigma));
            
            % 座標が行列の範囲内であるか確認
            if x >= 1 && x <= N && y >= 1 && y <= N
                % 重複しないようにする
                if matrix(x, y) == 0
                    matrix(x, y) = 1;
                    success = true;
                end
            end
            
            % 最大試行回数を超えた場合はリトライ
            attempts = attempts + 1;
            if attempts > 2000
                warning('Max attempts reached. Some points may not be placed.');
                break;
            end
        end
    end
end
