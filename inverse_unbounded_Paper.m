close all;
clear all;clc;

rng(0); 
GPU_num = 4;
gpuDevice(GPU_num); reset(gpuDevice(GPU_num)); executionEnvironment = 'gpu'; gpurng(0);

%サイズ
M = 10;     %uniformアレイの1辺の長さ
N = 41;    %アンテナ数

%計測回数
min_k = N^2*4;      % 開始値
max_k = N^2*12;     % 終了値
stride = N^2*2;     % 間隔
num_measurements = min_k:stride:max_k;

%実験回数
num_exp = 10;

%limit iteration
max_itr = 1e5;

%オリジナル画像
%{
img = imread('peppers.png');
img_gray =  double(rgb2gray(imresize(img, [N, N])));
obj = img_gray / max(img_gray(:));
obj = gpuArray(double(obj.*MyRect(N, N/1.5))); obj_name = 'peppers';
%}
%obj = gpuArray(double(MyRect(N,[N/2,N/7],[N/2,N/3]) + MyRect(N,[N/2,N/7],[N/2,2*N/3]))) ; obj_name = 'RomeTwo';
obj = gpuArray(double(MyRect(N, N/2))) ; obj_name = 'HalfSqr';

%サポート
sup_size = ceil(N/1.5);
sup =gpuArray(double(MyRect(N, sup_size)));

%アンテナ配置
%array = gpuArray(double(MyRect(N, M))); array_name = 'Uni' %for uniformアレイ
load('Costasarray_N41.mat') ; array = gpuArray(double(matrix)); array_name = 'Cos';

%ADAMのパラメタ
m_O = zeros(N,'double','gpuArray');
v_O = zeros(N,'double','gpuArray');
m_r = zeros(N,'double','gpuArray');
v_r = zeros(N,'double','gpuArray');

alpha = 1e-2;
beta_1 = 0.95;
beta_2 = 0.999;
epsilon = 1e-8;

%TVのパラメタ
v_TV_O =  ones(N,'double','gpuArray');
u_TV_O = zeros(N,'double','gpuArray');
%rho_O = 0; %TVなし
rho_O = 1e-3; %TVあり
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
    K = num_measurements(idx_K);

    rng(0); gpurng(0);
    RMSE_tmp_o = zeros(num_exp, 1); %平均・標準偏差計算
    RMSE_tmp_r = zeros(num_exp, 1);

    %SGDの設定
    num_epoch = 500;  
    batch_size = 2^5; %バッチサイズ
    num_itr = (ceil(K/batch_size))*num_epoch; %反復回数
    data_indice = randperm(K); %batch列を用意

    %位相シフトKパターン（N×N×K）
    %phi = array.*rand(N,N,K)*2*pi;
    phi = array.*rand(N,N,K,'double','gpuArray')*2*pi;
    
    %アンテナ配置×位相シフト（N×N×K）
    %A = array.*exp(1i*phi);
    A = array.*gpuArray(double(exp(1i*phi))); 

    for seed = 0:(num_exp-1) %位相バイアス複数通り試して標準偏差を取る
        %進捗を表示
        now = now + 1;
        progress = sprintf('K=%d,seed=%dを計算中（%d/%d)', K,seed,now,length(num_measurements)*num_exp);
        disp(progress);

        %位相バイアス（N×N）
        rng(seed); gpurng(seed);
        r = array.*rand(N,N,'double','gpuArray')*2*pi; 
        
        %PDの観測強度（K×1配列）
        S = zeros(1,K,'double','gpuArray');
        for batch_start = 1:batch_size:K
            batch_F = MyFFT2(A(:,:,batch_start:min(batch_start+batch_size-1, K)).*gpuArray(double(exp(1i*r))));
            S(batch_start:min(batch_start+batch_size -1, K)) = sum(abs(batch_F).^2.*obj.*sup, [1,2]);
        end
        S = reshape(S, [K,1]);
        S = awgn(S,60,'measured');
        clearvars batch_F

        %ここから逆問題
        figure(1);
        O_hat = ones(N,'double','gpuArray'); %Oの初期値（N×N）
        r_hat = array.*rand(N,'double','gpuArray')*2*pi; %rの初期値（N×N）
        batch_es = zeros(num_itr,1,'double','gpuArray');

        elapsed_times = zeros(floor(num_itr/100), 1);
        itr = 0; hundreds = 0;
        tic;

        for epoch = 1:num_epoch
            for batch_start = 1:batch_size:K
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
                dReLU_O = zeros(N, 'double','gpuArray');
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
                    if itr ~=num_itr
                      tic;
                    end 

                end

                if itr == max_itr
                    break;
                end

            end

            if itr == max_itr
                break;
            end
        end
        
        %ここから品質評価
        %objとO_hatの相互相関
        O_hat = real(O_hat); %念のため
        O_hat_flip = rot90(O_hat, 2);
        corr_map = real(MyIFFT2(MyFFT2(obj) .* MyFFT2(O_hat_flip)));
        
        %相互相関が最大となるindexを求め、O_hatのシフト量を求める
        [max_corr, max_corr_index] = max(corr_map(:));
        [max_corr_row, max_corr_col] = ind2sub(size(corr_map), max_corr_index);
        rows_shift = max_corr_row - ceil(N/2);
        cols_shift = max_corr_col - ceil(N/2);
        
        %O_hatのシフト量からr_hatのシフト量を算出しr_hatを補正、また0〜2piにラッピング 
        [meshx, meshy] = meshgrid(ceil(-(N-1)/2):ceil((N-1)/2), ceil(-(N-1)/2):ceil((N-1)/2));
        r_hat_shifted = (r_hat + 2*pi.*rows_shift.*meshy./N + 2*pi.*cols_shift.*meshx./N).*array;
        r_hat_flattened = wrapTo2Pi(angle(exp(1i.*r_hat_shifted)));
        
        %上記で求めたシフト量からO_hatを補正
        %（前処理1）support領域だけO_hatを切り取り（for support内の巡回）
        [row, col] = find(sup ~= 0);
        supportRegion = O_hat(row(1):row(end), col(1):col(end));
        
        %support領域のO_hatをシフト
        supportRegionShifted = circshift(supportRegion, [rows_shift, cols_shift]);
        
        %外側を0paddingしてsupport付き画像に戻す
        O_hat_shifted = zeros(N);
        O_hat_shifted(row(1):row(end), col(1):col(end)) = supportRegionShifted;
        
        %r_hatの定数加算量を推定し位相を補正
        dif_bias = wrapTo2Pi(r - r_hat_flattened);
        bias_shift = sum(dif_bias(:))/N;
        r_hat_flattened = wrapTo2Pi((r_hat_flattened() + bias_shift).*array);
        
        %RMSEの計算
        RMSE_o = sqrt(mean((O_hat_shifted(:) - obj(:)).^2));
        RMSE_r = sqrt(sum((r_hat_flattened(:) - r(:)).^2)/N);
        
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
        imagesc(O_hat); colormap gray; axis image; colorbar;
        title('Reconstructed amplitude');
        
        subplot(4,3,5)
        imagesc(r_hat); colormap gray; axis image; colorbar; clim([0, 2*pi]);
        title('Reconstructed phase bias');
        
        subplot(4,3,7)
        imagesc(O_hat_shifted); colormap gray; axis image; colorbar;
        title(['Corrected amplitude (RMSE=',num2str(RMSE_o, 4), ')']);
        
        subplot(4,3,8)
        imagesc(r_hat_flattened); colormap gray; axis image; colorbar; clim([0, 2*pi]);
        title(['Corrected phase bias (RMSE=',num2str(RMSE_r, 4), ')']);
        
        subplot(4,3,10)
        imagesc(corr_map); colormap gray; axis image; colorbar;
        title('Correlation map');
        
        subplot(4,3,[11,12])
        semilogy(batch_es(1:itr));
        title(['|S_{hat} - S|^2  (', num2str(sum(elapsed_times)/hundreds,4),'sec/100itr)']);

        drawnow();

        %結果を保存
        save_dir = sprintf('./figures/M2_%s_%s_N%d_K%d_sup%d/',array_name,obj_name,N,K,sup_size);
        mkdir(save_dir);
        filename_fig = sprintf('%s%d.fig',save_dir,seed);
        filename_png = sprintf('%s%d.png',save_dir,seed);
       
        %ファイルを保存
        savefig(filename_fig);
        print(filename_png, '-dpng', '-r300');

        finishMessage = sprintf('K=%d,seed=%dの結果を保存 (RMSE_o=%.4f, RMSE_r=%.4f)',K,seed,RMSE_o,RMSE_r);
        disp(finishMessage);

        %RMSEを保存
        RMSE_tmp_o(seed+1) = RMSE_o;
        RMSE_tmp_r(seed+1) = RMSE_r;
    end

    RMSEs_o(idx_K) = mean(RMSE_tmp_o);
    stds_o(idx_K) = std(RMSE_tmp_o);
    RMSEs_r(idx_K) = mean(RMSE_tmp_r);
    stds_r(idx_K) = std(RMSE_tmp_r);

    clearvars phi A data_indice
end

RMSE_path = './figures/RMSE/';
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

%???????
RMSE_fig = sprintf('%sM2_%s_%s_N%d_sup%d_numE%d.fig',RMSE_path,array_name,obj_name,N,sup_size, num_exp);
RMSE_png = sprintf('%sM2_%s_%s_N%d_sup%d_numE%d.png',RMSE_path,array_name,obj_name,N,sup_size, num_exp);
savefig(RMSE_fig);
print(RMSE_png, '-dpng', '-r300');

clearvars matrix img img_gray 






