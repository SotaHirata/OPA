close all;
clear all;clc;

rng(0); 
%GPU_num = 4;
%gpuDevice(GPU_num); reset(gpuDevice(GPU_num)); executionEnvironment = 'gpu'; gpurng(0);

M = 10;     %uniformアレイの1辺の長さ
N = M^2;    %アンテナ数

%計測回数
min_k = N^2*1;      % 開始値
max_k = N^2*4;     % 終了値
stride = N^2*1;     % 間隔
num_measurements = [min_k/10,min_k/2, min_k:stride:max_k];

%ランダムな位相バイアス（N×N）の枚数
num_phase_bias = 10;

%位相バイアス1つあたりの初期値数
num_inits = 5;

%AWGNのSN比
noiseLv = 30;

%limit iteration
max_itr = 1e4;

%SGDの設定
batch_size = 2^5; %バッチサイズ
num_epoch = 1000; %エポック数（今回はmax_itrで停止するので無関係. 十分大にしておく）

%サポート
%sup_size = ceil(N/1.5);
sup_size = N;
%sup =gpuArray(double(MyRect(N, sup_size)));
sup =MyRect(N, sup_size);
[row, col] = find(sup ~= 0); %サポート領域のインデックス

%オリジナル画像
%{
img = imread('peppers_color.png');
img_gray =  double(rgb2gray(imresize(img, [sup_size, sup_size])));
img_gray_normalized = img_gray / max(img_gray(:));
obj = zeros(N);
obj(row(1):row(end), col(1):col(end)) = img_gray_normalized;
obj = gpuArray(double(obj)); obj_name = 'peppers';
%}
%obj = gpuArray(double(MyRect(N,[N/2,N/7],[N/2,N/3]) + MyRect(N,[N/2,N/7],[N/2,2*N/3]))) ; obj_name = 'RomeTwo';
%obj = gpuArray(double(MyRect(N, N/2))) ; obj_name = 'HalfSqr';
obj = MyRect(N, N/2) ; obj_name = 'HalfSqr';

%アンテナ配置
%array = gpuArray(double(MyRect(N, M))); array_name = 'Uni'; %for uniformアレイ
array = MyRect(N, M); array_name = 'Uni'; %for uniformアレイ
%load('Costasarray_N101.mat') ; array = gpuArray(double(matrix)); array_name = 'Cos';
%load('Costasarray_N101.mat') ; array = matrix; array_name = 'Cos';

%位相バイアスのリスト
phase_biases = array.*(rand(N,N,num_phase_bias)*2*pi);

%初期値のリスト
O_hat_inits = rand(N,N,num_inits); %Oの初期値のリスト
r_hat_inits = array.*rand(N,N,num_inits)*2*pi; %rの初期値のリスト

%ADAMのパラメタ
m_O = zeros(N);
v_O = zeros(N);
m_r = zeros(N);
v_r = zeros(N);
alpha = 2e-2;
beta_1 = 0.95;
beta_2 = 0.999;
epsilon = 1e-8;

%TVのパラメタ
v_TV_O =  ones(N);
u_TV_O = zeros(N);
%rho_O = 0; %TVなし
rho_O = 1e-2; %TVあり
tv_th = 1e-2;
tv_tau = 0.05;
tv_iter = 4; %TVの反復数

%非負ペナルティ
mu = 1e8; 

%進捗表示用
now = 0;
%RMSEグラフ描画用
RMSEs_o = zeros(length(num_measurements), 1);
stds_o = zeros(length(num_measurements), 1);
RMSEs_r = zeros(length(num_measurements), 1);
stds_r = zeros(length(num_measurements), 1);

for idx_K = 1:length(num_measurements)    %計測回数Kループ
    %計測回数を設定
    K = num_measurements(idx_K);
    %ミニバッチ列を用意
    data_indice = randperm(K); 

    %位相シフトKパターン（N×N×K）を設定
    phi = array.*rand(N,N,K)*2*pi;
    %phi = array.*rand(N,N,K,'double','gpuArray')*2*pi;
    
    %アンテナ配置×位相シフト（N×N×K）
    A = array.*exp(1i*phi);
    %A = array.*gpuArray(double(exp(1i*phi))); 

    %RMSEの平均・標準偏差計算のための記憶用変数の初期化
    RMSE_tmp_o = zeros(num_phase_bias, 1); 
    RMSE_tmp_r = zeros(num_phase_bias, 1); 

    %位相バイアスnum_phase_bias通りにたいして再構成を試す
    for seed = 1:num_phase_bias 
        %位相バイアス（N×N）を設定
        r = phase_biases(:,:,seed);
        
        %順伝播:PDの観測強度（K×1）を計算
        S = zeros(1,K);
        for batch_start = 1:batch_size:K
            %batch_F = MyFFT2(A(:,:,batch_start:min(batch_start+batch_size-1, K)).*gpuArray(double(exp(1i*r))));
            batch_F = MyFFT2(A(:,:,batch_start:min(batch_start+batch_size-1, K)).*exp(1i*r));
            S(batch_start:min(batch_start+batch_size -1, K)) = sum(abs(batch_F).^2.*obj.*sup, [1,2]);
        end
        S = reshape(S, [K,1]);
        S = awgn(S,noiseLv,'measured');
        clearvars batch_F

        %逆問題
        
        %最良推定値の保存変数の初期化
        O_hat_best = zeros(N);
        r_hat_best = zeros(N);
        batch_es_best = zeros(max_itr,1);
        batch_es_best(max_itr) = 1e10; %十分大きく設定しておく

        %初期値をnum_inits通り降って、最良のRMSE_rのケースを探索
        for trial = 1:num_inits
            %進捗を表示
            now = now + 1;
            progress = sprintf('K=%d,seed=%d,trial=%d を計算中（%d/%d) loss_best=%.3e', K,seed,trial,now,length(num_measurements)*num_phase_bias*num_inits,batch_es_best(max_itr));
            disp(progress);

            figure(1);
            O_hat = O_hat_inits(:,:,trial);
            r_hat = r_hat_inits(:,:,trial);
            batch_es = zeros(max_itr,1);

            elapsed_times = zeros(floor(max_itr/100), 1);
            itr = 0; hundreds = 0;
            tic;
    
            for epoch = 1:num_epoch
                for batch_start = 1:batch_size:K %ミニバッチ
                    itr = itr + 1;
            
                    batch_idx = data_indice(batch_start:min(batch_start+batch_size-1, K));
                    batch_A = A(:,:,batch_idx);
                    batch_F = MyFFT2(batch_A.*exp(1i*r_hat));
                    batch_S_hat = reshape(sum(abs(batch_F).^2.*O_hat, [1,2]), [length(batch_idx),1]);
                    batch_e = batch_S_hat - S(batch_idx);
                    batch_es(itr) = mean(abs(batch_e).^2, 'all');
            
                    %非負ペナルティ
                    ReLU_O = -O_hat;
                    ReLU_O(O_hat>0) = 0;
                    dReLU_O = zeros(N);
                    dReLU_O(O_hat<0) = -1;
            
                    %O,rの勾配
                    st_O = 2*sum(abs(batch_F).^2.*reshape(batch_e, [1,1,length(batch_idx)]), 3) + 2.*rho_O.*(O_hat - (v_TV_O - u_TV_O)) + 2*mu*dReLU_O.*ReLU_O;
                    st_r = 2*(-1i*exp(-1i*r_hat)).*sum(2*conj(batch_A).*MyIFFT2(batch_F.*O_hat.*reshape(batch_e, [1,1,length(batch_idx)])),3);
                        
                    %Adam
                    [st_O, m_O, v_O] = Adam_func(st_O,m_O,v_O,itr,alpha,beta_1,beta_2,epsilon);
                    [st_r, m_r, v_r] = Adam_func(st_r,m_r,v_r,itr,alpha,beta_1,beta_2,epsilon);
                   
                    %O,rの更新
                    O_hat = (O_hat - st_O).*sup; %Oの更新式 
                    r_hat = r_hat - real(st_r); %rの更新式
                
                    %v,uの更新
                    v_TV_O = reshape(MyTVpsi_ND(O_hat + u_TV_O, tv_th, tv_tau, tv_iter, [N, N]), [N, N]);
                    u_TV_O = u_TV_O + (O_hat - v_TV_O);
                
                    if rem(itr, 100)==0    %描画
                        hundreds = hundreds + 1;
    
                        subplot(3,2,1)
                        imagesc(obj); colormap gray; axis image; colorbar;
                        title('Original object');
                
                        subplot(3,2,2)
                        imagesc(r); colormap gray; axis image; colorbar;
                        title('Original phase bias');
                
                        subplot(3,2,3)
                        imagesc(real(O_hat)); colormap gray; axis image; colorbar;
                        title(['Reconstructed image ( itr=',num2str(itr), ' )']);
                
                        subplot(3,2,4)
                        imagesc(r_hat); colormap gray; axis image; colorbar;
                        title(['Reconstructed phase bias  ( itr=',num2str(itr), ' )']);
                
                        subplot(3,2,[5,6])
                        semilogy(batch_es(1:itr));
                        title(['|S_{hat} - S|^2 (', num2str(sum(elapsed_times)/hundreds,4),'sec/100itr) ','(batchsize=',num2str(batch_size),')']);
                
                        drawnow(); 
                
                        %イテレーションごとの所要時間
                        elapsed_time = toc;
                        %fprintf('イテレーション %d の経過時間: %f 秒\n', itr, elapsed_time);
                        elapsed_times(hundreds) = elapsed_time;
                        if itr ~=max_itr
                          tic;
                        end 
    
                    end %描画終わり
    
                    if itr == max_itr %max_itrに達したとき更新終了
                        break;
                    end

                end %ミニバッチ終わり
    
                if itr == max_itr %max_itrに達したとき更新終了
                    break;
                end

            end 

            %最終的なロス（forミニバッチ）が最小の場合、各推定値とロス推移を記録。
            if batch_es(itr) < batch_es_best(itr)
                O_hat_best = O_hat;
                r_hat_best = r_hat;
                batch_es_best(1:itr) = batch_es(1:itr);
            end
        end %初期値を振ってのtrialループ終了

        %ここから品質評価
        O_hat_best = real(O_hat_best); %念のため

        %サポート上のobjとO_hat_bestの相互相関
        O_hat_onSup = O_hat_best(row(1):row(end), col(1):col(end));
        obj_onSup = obj(row(1):row(end), col(1):col(end));
        O_hat_onSup_flip = rot90(O_hat_onSup, 2);
        corr_map = real(MyIFFT2(MyFFT2(obj_onSup) .* MyFFT2(O_hat_onSup_flip)));
        
        %相互相関が最大となるindexを求め、サポート上のO_hat_bestのシフト量を求める
        [max_corr, max_corr_index] = max(corr_map(:));
        [max_corr_row, max_corr_col] = ind2sub(size(corr_map), max_corr_index);
        rows_shift = max_corr_row - ceil(length(corr_map)/2) ;
        cols_shift = max_corr_col - ceil(length(corr_map)/2) ;

        %8近傍シフト時の最良補正値の保存変数の初期化
        RMSE_o_best = 0;
        RMSE_r_best = 1000; 
        O_hat_shifted_best = zeros(N);
        exp_r_hat_corrected_best = exp(1i*ones(N));

        %8近傍でRMSE_rが最小となるシフト量を探索
        for row_add = -1:1
            for col_add = -1:1
                %row_shift, col_shiftを8近傍にシフト
                rows_shift_added = rows_shift + row_add;
                cols_shift_added = cols_shift + col_add;

                %サポート上のO_hatをシフト
                O_hat_onSup_shift = circshift(O_hat_onSup, [rows_shift_added, cols_shift_added]);
        
                %外側を0paddingしてsupport付き画像に戻す
                O_hat_shifted = zeros(N);
                O_hat_shifted(row(1):row(end), col(1):col(end)) = O_hat_onSup_shift;
                        
                %O_hatのシフト量からr_hatのシフト量を算出しr_hatを補正 
                [meshx, meshy] = meshgrid(ceil(-(N-1)/2):ceil((N-1)/2), ceil(-(N-1)/2):ceil((N-1)/2));
                r_hat_shifted = (r_hat_best + 2*pi.*rows_shift_added.*meshy./N + 2*pi.*cols_shift_added.*meshx./N).*array;
                
                %r_hatのオフセット量を推定し位相を補正
                exp_dif_bias = exp(1i*(r - r_hat_shifted));
                bias_offset = sum(angle(exp_dif_bias(:)))/N;
                exp_r_hat_corrected = exp(1i*(r_hat_shifted +bias_offset).*array);
                
                %RMSEの計算
                RMSE_o = sqrt(mean((O_hat_shifted(:) - obj(:)).^2));
                RMSE_r = sqrt(sum(abs(exp_r_hat_corrected(:) - exp(1i*r(:))).^2)/N);

                %RMSE_rが最小の時の各補正値を保持
                if RMSE_r < RMSE_r_best
                    RMSE_o_best = RMSE_o;
                    RMSE_r_best = RMSE_r;
                    O_hat_shifted_best = O_hat_shifted;
                    exp_r_hat_corrected_best = exp_r_hat_corrected;
                end
            end
        end


        % 結果の表示
        figure(2);
        gcf.Position = [714 91 818 775];
        subplot(4,3,1)
        imagesc(obj); colormap gray; axis image; colorbar;
        title('Original object amplitude');
        
        subplot(4,3,2)
        imagesc(r); colormap gray; axis image; colorbar; clim([0, 2*pi]);
        title('Original phase bias');
        
        subplot(4,3,4)
        imagesc(O_hat_best); colormap gray; axis image; colorbar;
        title('Reconstructed amplitude');
        
        subplot(4,3,5)
        imagesc(r_hat_best); colormap gray; axis image; colorbar; clim([0, 2*pi]);
        title('Reconstructed phase bias');
        
        subplot(4,3,7)
        imagesc(O_hat_shifted_best); colormap gray; axis image; colorbar;
        title(['Corrected amplitude (RMSE=',num2str(RMSE_o_best, 4), ')']);
        
        subplot(4,3,8)
        imagesc(wrapTo2Pi(angle(exp_r_hat_corrected_best))); colormap gray; axis image; colorbar; clim([0, 2*pi]);
        title(['Corrected phase bias (RMSE=',num2str(RMSE_r_best, 4), ')']);
        
        subplot(4,3,10)
        imagesc(corr_map); colormap gray; axis image; colorbar;
        title('Correlation map');
        
        subplot(4,3,[11,12])
        semilogy(batch_es_best(1:itr));
        title('|S_{hat} - S|^2');

        drawnow();

        %結果を保存
        save_dir = sprintf('./figures6/M2_%s_%s_N%d_K%d_sup%d_noise%d/',array_name,obj_name,N,K,sup_size,noiseLv);
        mkdir(save_dir);
        filename_fig = sprintf('%s%d.fig',save_dir,seed);
        filename_png = sprintf('%s%d.png',save_dir,seed);
       
        %ファイルを保存
        savefig(filename_fig);
        print(filename_png, '-dpng', '-r300');

        finishMessage = sprintf('K=%d,seed=%dの結果を保存 (RMSE_o_best=%.4f, RMSE_r_best=%.4f)',K,seed,RMSE_o_best,RMSE_r_best);
        disp(finishMessage);

        %RMSEを保存
        RMSE_tmp_o(seed) = RMSE_o_best;
        RMSE_tmp_r(seed) = RMSE_r_best;
    end

    RMSEs_o(idx_K) = mean(RMSE_tmp_o);
    stds_o(idx_K) = std(RMSE_tmp_o);
    RMSEs_r(idx_K) = mean(RMSE_tmp_r);
    stds_r(idx_K) = std(RMSE_tmp_r);

    clearvars phi A data_indice
end

RMSE_path = './figures6/RMSE/';
mkdir(RMSE_path);

figure(3);
subplot(1,2,1)
errorbar(num_measurements,RMSEs_o,stds_o);
title('RMSE of object');
xlabel('Number of measurements');
ylabel('Reconstruction RMSE');

subplot(1,2,2)
errorbar(num_measurements,RMSEs_r,stds_r);
title('RMSE of phase bias');
xlabel('Number of measurements');
ylabel('Reconstruction RMSE');

%RMSEのグラフを保存
RMSE_fig = sprintf('%sM2_%s_%s_N%d_sup%d_noise%d_numE%d.fig',RMSE_path,array_name,obj_name,N,sup_size,noiseLv,num_phase_bias);
RMSE_png = sprintf('%sM2_%s_%s_N%d_sup%d_noise%d_numE%d.png',RMSE_path,array_name,obj_name,N,sup_size,noiseLv,num_phase_bias);
savefig(RMSE_fig);
print(RMSE_png, '-dpng', '-r300');

clearvars matrix img img_gray 
