%% 初期設定
close all; clear all; clc; rng(0); 
%GPU_num = 4; %gpuDevice(GPU_num); reset(gpuDevice(GPU_num)); executionEnvironment = 'gpu'; gpurng(0);
delete(gcp('nocreate')); poolobj = parpool('Threads'); %並列処理用

M = 10;      %uniformアレイの1辺の長さ
N = 100;     %アンテナの1辺の長さ
%K = N^2*4;   %計測回数
K = N^2*8;   %計測回数
maskD = N/8; %PD直前のピンホールの直径

%Gaussianアレイの設定
num_antenna = N;
array_name = sprintf('GaussAntn%d',num_antenna);
%Gaussianアレイのシグマのパターン
sigmas = [1.5,2.5,5,7.5,10,12.5,15];
%sigmas = [5];
sigmas_len = length(sigmas);

%あるシグマのGaussianアレイのパターン数
num_Gauss_trial = 10;

%位相バイアス1つあたりの初期値数
num_inits = 5;

%AWGNのSN比
noiseLv = 40;

%最大反復数
max_itr = 1.5e4;

%SGDの設定
batch_size = 2^5; %バッチサイズ
num_epoch = 1000; %エポック数（今回はmax_itrで停止するので無関係. 十分大にしておく）
data_indice = randperm(K); %ミニバッチ列を用意
    
%サポート
sup_size = N;
sup = MyRect(N, sup_size);
[row, col] = find(sup ~= 0); %サポート領域のインデックス

%ピンホール
mask = MyCirc(N, maskD); 

%オリジナル画像
img = imread('peppers_color.png');
img_gray =  double(rgb2gray(imresize(img, [sup_size, sup_size])));
img_gray_normalized = img_gray / max(img_gray(:));
img_gray_normalized = img_gray_normalized.*exp(1i*(pi*(0.5+rot90(img_gray_normalized))));
obj = zeros(N);
obj(row(1):row(end), col(1):col(end)) = img_gray_normalized; obj_name = 'peppers';

%ADAMのパラメタ
%alpha = 3e-2; %計測回数N^2x8, maskD=N/8, sigma=7.5のベスト
alpha = 2.5e-2;
beta_1 = 0.98;
beta_2 = 0.999;
epsilon = 1e-8;

%後で消す
%{
%rho_O = 5e1; %計測回数N^2x8, maskD=N/8, sigma=7.5でのそこそこの成功値
%rho_O = 3e1; %計測回数N^2x8, maskD=N/8, sigma=7.5でのそこそこの成功値
rho_O = 5e2;
tv_th = 1e-2;
tv_tau = 0.05;
tv_iter = 5; 
%}

%
%% TVのパラメタのグリッドサーチ
%（もしかして）強度物体モデルより1e4くらい小さいほうがいい説？
rho_Os = [1,5,10,50,100].*1e1; 
num_rho_Os = length(rho_Os);
tv_th = 1e-2;
tv_tau = 0.05;
tv_iter = 5; %TVの反復数


%アンテナ数ごとにチューニングされた最適なrho_Oを保持する変数
rho_O_tuned = zeros(sigmas_len,1);

for idx_sigma = 1:sigmas_len %シグマごとに最適なrho_Oを決める
    %σを設定
    sigma = sigmas(idx_sigma);
    %アンテナ配置
    array = Gaussianarray_gen(N,num_antenna,sigma);
    %位相シフトKパターン（N×N×K）を設定
    phi = array.*rand(N,N,K)*2*pi;
    %アンテナ配置×位相シフト（N×N×K）
    A = array.*exp(1i*phi);
    %位相バイアス（N×N）を設定
    r = array.*(rand(N)*2*pi);

    %順伝播:PDの観測強度（K×1）を計算
    S = zeros(1,K);
    for batch_start = 1:batch_size:K
        S(batch_start:min(batch_start+batch_size -1, K)) = sum(abs(MyIFFT2(MyFFT2(A(:,:,batch_start:min(batch_start+batch_size -1, K)).*exp(1i*r)).*obj.*sup)).^2.*mask, [1,2]); 
    end
    S = reshape(S, [K,1]);
    S = awgn(S,noiseLv,'measured');
    clearvars batch_F

    %各rho_Oに対するRMSE_oを記録する変数
    RMSE_TVs = zeros(num_rho_Os,1);
    %初期値を統一
    O_hat_init = rand(N).*exp(1i.*2*pi*rand(N));
    r_hat_init = array.*rand(N).*2*pi;
    
    %for idx_rho = 1:num_rho_Os %rho_Oを切り替えて最適なrho_Oを探索
    parfor idx_rho = 1:num_rho_Os %rho_Oを切り替えて最適なrho_Oを探索
        %rho_Oを設定
        rho_O = rho_Os(idx_rho);
        
        %逆問題
        %figure(idx_rho);
        O_hat = O_hat_init;
        r_hat = r_hat_init;
        batch_es = zeros(max_itr,1);

        %ADAMの初期化
        m_O = zeros(N);
        v_O = zeros(N);
        m_r = zeros(N);
        v_r = zeros(N);

        %TVの初期化
        v_TV_O =  ones(N);
        u_TV_O = zeros(N);

        %elapsed_times = zeros(floor(max_itr/100), 1);
        itr = 0; %hundreds = 0;
        %tic;

        for epoch = 1:num_epoch
            for batch_start = 1:batch_size:K %ミニバッチ
                itr = itr + 1;
        
                batch_idx = data_indice(batch_start:min(batch_start+batch_size-1, K));
                batch_A = A(:,:,batch_idx);
                batch_F = MyFFT2(batch_A.*exp(1i*r_hat));
                batch_I = MyIFFT2(batch_F.*O_hat); 
                batch_S_hat = reshape(sum(abs(batch_I).^2.*mask, [1,2]),[length(batch_idx),1]); 
                batch_e = batch_S_hat - S(batch_idx);
                batch_es(itr) = mean(abs(batch_e).^2, 'all');
        
                %O,rの勾配
                st_O = 2*sum(2*conj(batch_F).*MyFFT2(batch_I.*mask.*reshape(batch_e,[1,1,length(batch_idx)])),3) + 2.*rho_O.*(O_hat - (v_TV_O - u_TV_O));
                st_r = 2*(-1i*exp(-1i*r_hat)).*sum(2*conj(batch_A).*MyIFFT2(conj(O_hat).*MyFFT2(batch_I.*mask.*reshape(batch_e,[1,1,length(batch_idx)]))),3);
                    
                %Adam
                [st_O, m_O, v_O] = Adam_func(st_O,m_O,v_O,itr,alpha,beta_1,beta_2,epsilon);
                [st_r, m_r, v_r] = Adam_func(st_r,m_r,v_r,itr,alpha,beta_1,beta_2,epsilon);
               
                %O,rの更新
                O_hat = (O_hat - st_O).*sup; %Oの更新式 
                r_hat = r_hat - real(st_r); %rの更新式
            
                %v,uの更新
                v_TV_O_re = reshape(MyTVpsi_ND(real(O_hat + u_TV_O), tv_th, tv_tau, tv_iter, [N, N]), [N, N]);
                v_TV_O_im = reshape(MyTVpsi_ND(imag(O_hat + u_TV_O), tv_th, tv_tau, tv_iter, [N, N]), [N, N]);
                v_TV_O = v_TV_O_re + 1i .*v_TV_O_im;
                u_TV_O = u_TV_O + (O_hat - v_TV_O);
            
                %{
                if rem(itr, 100)==0    %描画
                    hundreds = hundreds + 1;

                    subplot(3,3,1)
                    imagesc(abs(obj)); colormap gray; axis image; colorbar;
                    title('Original amp.');

                    subplot(3,3,2)
                    imagesc(wrapTo2Pi(angle(obj))); colormap gray; axis image; colorbar;
                    title('Original phase');
            
                    subplot(3,3,3)
                    imagesc(r); colormap gray; axis image; colorbar;
                    title('Original phase bias');
            
                    subplot(3,3,4)
                    imagesc(abs(O_hat)); colormap gray; axis image; colorbar;
                    title(['Reconstructed amp. (itr=',num2str(itr), ')']);

                    subplot(3,3,5)
                    imagesc(wrapTo2Pi(angle(O_hat))); colormap gray; axis image; colorbar;
                    title(['Reconstructed phase (itr=',num2str(itr), ')']);
            
                    subplot(3,3,6)
                    imagesc(r_hat); colormap gray; axis image; colorbar;
                    title(['Reconstructed phase bias  (itr=',num2str(itr), ')']);
            
                    subplot(3,3,[7,8,9])
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
                %}

                if itr == max_itr %max_itrに達したとき更新終了
                    break;
                end

            end %ミニバッチ終わり

            if itr == max_itr %max_itrに達したとき更新終了
                break;
            end
        end %エポック終わり

        %ここから品質評価
        %サポート上のobjの絶対値とO_hat_bestの絶対値の相互相関
        O_hat_onSup = O_hat(row(1):row(end), col(1):col(end));
        obj_onSup = obj(row(1):row(end), col(1):col(end));
        O_hat_onSup_flip = rot90(O_hat_onSup, 2);
        corr_map = real(MyIFFT2(MyFFT2(abs(obj_onSup)) .* MyFFT2(abs(O_hat_onSup_flip))));
        
        %相互相関が最大となるindexを求め、サポート上のO_hat_bestのシフト量を求める
        [max_corr, max_corr_index] = max(corr_map(:));
        [max_corr_row, max_corr_col] = ind2sub(size(corr_map), max_corr_index);
        rows_shift = max_corr_row - ceil(length(corr_map)/2) ;
        cols_shift = max_corr_col - ceil(length(corr_map)/2) ;

        %24近傍シフト時の最良補正値の保存変数の初期化
        RMSE_o_best = 0;
        RMSE_o_abs_best = 0;
        RMSE_o_angle_best = 0;
        RMSE_r_best = 1000; 
        O_hat_corrected_best = zeros(N);
        exp_r_hat_corrected_best = exp(1i*ones(N));
        row_add_best = 0;
        col_add_best = 0;
        dif_bias_best = 0;

        %24近傍でRMSE_rが最小となるシフト量を探索
        for row_add = -2:2
            for col_add = -2:2
                %row_shift, col_shiftを24近傍にシフト
                rows_shift_added = rows_shift + row_add;
                cols_shift_added = cols_shift + col_add;

                %サポート上のO_hatをシフト
                O_hat_onSup_shift = circshift(O_hat_onSup, [rows_shift_added, cols_shift_added]);
        
                %外側を0paddingしてsupport付き画像に戻す
                O_hat_shifted = zeros(N);
                O_hat_shifted(row(1):row(end), col(1):col(end)) = O_hat_onSup_shift;
                        
                %O_hatのシフト量からr_hatのシフト量を算出しr_hatを補正 
                [meshx, meshy] = meshgrid(ceil(-(N-1)/2):ceil((N-1)/2), ceil(-(N-1)/2):ceil((N-1)/2));
                r_hat_shifted = (r_hat + 2*pi.*rows_shift_added.*meshy./N + 2*pi.*cols_shift_added.*meshx./N).*array;
                
                %angle(O_hat_shifted)のオフセット量を推定し位相を補正
                dif_angle = wrapTo2Pi(angle(obj)) - wrapTo2Pi(angle(O_hat_shifted));
                %for [0,2pi]
                dif_angle_1 =  wrapTo2Pi(dif_angle);
                angle_offset_1 = sum(dif_angle_1(:))/nnz(sup);
                O_hat_corrected_1 = O_hat_shifted.*exp(1i*angle_offset_1);
                %for [-pi,pi]
                dif_angle_2 =  wrapToPi(dif_angle);
                angle_offset_2 = sum(dif_angle_2(:))/nnz(sup);
                O_hat_corrected_2 = O_hat_shifted.*exp(1i*angle_offset_2);

                %r_hatのオフセット量を推定し位相を補正
                dif_bias = r - r_hat_shifted;
                %for [0,2pi]
                dif_bias_1 =  wrapTo2Pi(dif_bias);
                bias_offset_1 = sum(dif_bias_1(:))/num_antenna;
                exp_r_hat_corrected_1 = exp(1i*(r_hat_shifted +bias_offset_1).*array);
                %for [-pi,pi]
                dif_bias_2 =  wrapToPi(dif_bias);
                bias_offset_2 = sum(dif_bias_2(:))/num_antenna;
                exp_r_hat_corrected_2 = exp(1i*(r_hat_shifted +bias_offset_2).*array);

                %RMSEの計算
                RMSE_o_1 = sqrt(mean(abs(O_hat_corrected_1(:) - obj(:)).^2));
                RMSE_o_2 = sqrt(mean(abs(O_hat_corrected_2(:) - obj(:)).^2));

                RMSE_o_abs = sqrt(mean((abs(O_hat_corrected_1(:)) - abs(obj(:))).^2));
                RMSE_o_angle_1 = sqrt(mean((wrapTo2Pi(angle(O_hat_corrected_1(:))) - wrapTo2Pi(angle(obj(:)))).^2));
                RMSE_o_angle_2 = sqrt(mean((wrapTo2Pi(angle(O_hat_corrected_2(:))) - wrapTo2Pi(angle(obj(:)))).^2));
                
                RMSE_r_1 = sqrt(sum(abs(exp_r_hat_corrected_1(:) - exp(1i*r(:))).^2)/num_antenna);
                RMSE_r_2 = sqrt(sum(abs(exp_r_hat_corrected_2(:) - exp(1i*r(:))).^2)/num_antenna);
                
                if RMSE_o_1 < RMSE_o_2
                    RMSE_o = RMSE_o_1;
                    RMSE_o_angle = RMSE_o_angle_1;
                    O_hat_corrected = O_hat_corrected_1;
                else
                    RMSE_o = RMSE_o_2;
                    RMSE_o_angle = RMSE_o_angle_2;
                    O_hat_corrected = O_hat_corrected_2;
                end

                if RMSE_r_1 < RMSE_r_2
                    RMSE_r = RMSE_r_1;
                    exp_r_hat_corrected = exp_r_hat_corrected_1;
                else
                    RMSE_r = RMSE_r_2;
                    exp_r_hat_corrected = exp_r_hat_corrected_2;
                end

                %RMSE_rが最小の時の各補正値を保持
                if RMSE_r < RMSE_r_best
                    RMSE_o_best = RMSE_o;
                    RMSE_o_abs_best =  RMSE_o_abs;
                    RMSE_o_angle_best = RMSE_o_angle;
                    RMSE_r_best = RMSE_r;
                    O_hat_corrected_best = O_hat_corrected;
                    exp_r_hat_corrected_best = exp_r_hat_corrected;
                    row_add_best = row_add;
                    col_add_best = col_add;
                    dif_angle_best = dif_angle;
                    dif_bias_best = dif_bias;
                end
            end
        end

        RMSE_TVs(idx_rho) = RMSE_o_best;

        %進捗を表示
        progress = sprintf('sigma=%.1f (%d/%d),rho_o=%.2e(%d/%d)のとき,RMSE:o_abs=%.4f,o_angle=%.4f,o_total=%.4f',sigma,idx_sigma,sigmas_len,rho_O,idx_rho,num_rho_Os,RMSE_o_abs_best,RMSE_o_angle_best,RMSE_o_best);
        disp(progress);

    end %rho_Oを切り替えてのループ終わり

    [min_RMSE, min_idx_rho] = min(RMSE_TVs);
    rho_O_tuned(idx_sigma) = rho_Os(min_idx_rho);

    %進捗を表示
    progress = sprintf('sigma=%.1f (%d/%d)のチューニング結果:rho_O=%.2e,RMSE_o=%.4f',sigma,idx_sigma,sigmas_len,rho_Os(min_idx_rho),min_RMSE);
    disp(progress);

end %シグマごとのrho_Oのチューニング終了
%

%% ハイパーパラメータ決定後の検証

%rho_O_tuned = [rho_O]; %後で消す

%RMSEグラフ描画用
%複素版RMSE
RMSEs_o = zeros(sigmas_len, 1);
stds_rmse_o = zeros(sigmas_len, 1);
%振幅と位相それぞれのRMSE
RMSEs_o_abs = zeros(sigmas_len, 1);
stds_rmse_o_abs = zeros(sigmas_len, 1);
RMSEs_o_angle = zeros(sigmas_len, 1);
stds_rmse_o_angle = zeros(sigmas_len, 1);
%位相バイアスのRMSE
RMSEs_r = zeros(sigmas_len, 1);
stds_r = zeros(sigmas_len, 1);

for idx_sigma = 1:sigmas_len     %シグマを切り替えてループ
    %シグマを設定
    sigma = sigmas(idx_sigma);
    %ハイパーパラメータを設定
    rho_O = rho_O_tuned(idx_sigma);

    %RMSEの平均・標準偏差計算のための記憶用変数の初期化
    RMSE_tmp_o = zeros(num_Gauss_trial, 1);
    RMSE_tmp_o_abs = zeros(num_Gauss_trial, 1);
    RMSE_tmp_o_angle = zeros(num_Gauss_trial, 1);
    RMSE_tmp_r = zeros(num_Gauss_trial, 1); 

    %アンテナ配置seedをnum_Gauss_trial通り切り替えてループ
    for seed = 1:num_Gauss_trial
        %アンテナ配置
        array = Gaussianarray_gen(N,num_antenna,sigma);
        %位相シフトKパターン（N×N×K）を設定
        phi = array.*rand(N,N,K)*2*pi;
        %アンテナ配置×位相シフト（N×N×K）
        A = array.*exp(1i*phi);
        %位相バイアス（N×N）を設定
        r = array.*(rand(N)*2*pi);

        %初期値のリスト
        O_hat_inits = rand(N,N,num_inits).*exp(1i.*2*pi*rand(N,N,num_inits)); %Oの初期値のリスト
        r_hat_inits = array.*rand(N,N,num_inits)*2*pi; %rの初期値のリスト
       
        %順伝播:PDの観測強度（K×1）を計算
        S = zeros(1,K);
        for batch_start = 1:batch_size:K
            S(batch_start:min(batch_start+batch_size -1, K)) = sum(abs(MyIFFT2(MyFFT2(A(:,:,batch_start:min(batch_start+batch_size -1, K)).*exp(1i*r)).*obj.*sup)).^2.*mask, [1,2]); 
        end
        S = reshape(S, [K,1]);
        S = awgn(S,noiseLv,'measured');
        clearvars batch_F

        %逆問題

        %各初期値に対する推定値の記録用配列
        O_hat_results = zeros(N,N,num_inits);
        r_hat_results = zeros(N,N,num_inits);
        batch_es_results = zeros(max_itr,num_inits);

        %初期値（O_hat, r_hat）をnum_inits通り降って、最良のRMSE_rのケースを探索
        %for trial = 1:num_inits
        parfor trial = 1:num_inits
            %進捗を表示
            progress = sprintf('sigma=%.1f (%d/%d),Gaussアレイseed(%d/%d),初期値trial(%d/%d)を計算中',sigma,idx_sigma,sigmas_len,seed,num_Gauss_trial,trial,num_inits);
            disp(progress);
            
            %figure(trial);
            O_hat = O_hat_inits(:,:,trial);
            r_hat = r_hat_inits(:,:,trial);
            batch_es = zeros(max_itr,1);

            %elapsed_times = zeros(floor(max_itr/100), 1);
            itr = 0; %hundreds = 0;
            %tic;
    
            %ADAMの初期化
            m_O = zeros(N);
            v_O = zeros(N);
            m_r = zeros(N);
            v_r = zeros(N);

            %TVの初期化
            v_TV_O =  ones(N);
            u_TV_O = zeros(N);

            for epoch = 1:num_epoch
                for batch_start = 1:batch_size:K %ミニバッチ
                    itr = itr + 1;
        
                    batch_idx = data_indice(batch_start:min(batch_start+batch_size-1, K));
                    batch_A = A(:,:,batch_idx);
                    batch_F = MyFFT2(batch_A.*exp(1i*r_hat));
                    batch_I = MyIFFT2(batch_F.*O_hat); 
                    batch_S_hat = reshape(sum(abs(batch_I).^2.*mask, [1,2]),[length(batch_idx),1]); 
                    batch_e = batch_S_hat - S(batch_idx);
                    batch_es(itr) = mean(abs(batch_e).^2, 'all');
            
                    %O,rの勾配
                    st_O = 2*sum(2*conj(batch_F).*MyFFT2(batch_I.*mask.*reshape(batch_e,[1,1,length(batch_idx)])),3) + 2.*rho_O.*(O_hat - (v_TV_O - u_TV_O));
                    st_r = 2*(-1i*exp(-1i*r_hat)).*sum(2*conj(batch_A).*MyIFFT2(conj(O_hat).*MyFFT2(batch_I.*mask.*reshape(batch_e,[1,1,length(batch_idx)]))),3);
                        
                    %Adam
                    [st_O, m_O, v_O] = Adam_func(st_O,m_O,v_O,itr,alpha,beta_1,beta_2,epsilon);
                    [st_r, m_r, v_r] = Adam_func(st_r,m_r,v_r,itr,alpha,beta_1,beta_2,epsilon);
                   
                    %O,rの更新
                    O_hat = (O_hat - st_O).*sup; %Oの更新式 
                    r_hat = r_hat - real(st_r); %rの更新式
                
                    %v,uの更新
                    v_TV_O_re = reshape(MyTVpsi_ND(real(O_hat + u_TV_O), tv_th, tv_tau, tv_iter, [N, N]), [N, N]);
                    v_TV_O_im = reshape(MyTVpsi_ND(imag(O_hat + u_TV_O), tv_th, tv_tau, tv_iter, [N, N]), [N, N]);
                    v_TV_O = v_TV_O_re + 1i .*v_TV_O_im;
                    u_TV_O = u_TV_O + (O_hat - v_TV_O);
                
                    %{
                    if rem(itr, 100)==0    %描画
                        hundreds = hundreds + 1;

                        subplot(3,3,1)
                        imagesc(abs(obj)); colormap gray; axis image; colorbar;
                        title('Original amp.');
    
                        subplot(3,3,2)
                        imagesc(wrapTo2Pi(angle(obj))); colormap gray; axis image; colorbar; clim([0,2*pi]);
                        title('Original phase');
                
                        subplot(3,3,3)
                        imagesc(r); colormap gray; axis image; colorbar;
                        title('Original phase bias');
                
                        subplot(3,3,4)
                        imagesc(abs(O_hat)); colormap gray; axis image; colorbar;
                        title(['Reconstructed amp. (itr=',num2str(itr), ')']);
    
                        subplot(3,3,5)
                        imagesc(wrapTo2Pi(angle(O_hat))); colormap gray; axis image; colorbar; clim([0,2*pi]);
                        title(['Reconstructed phase (itr=',num2str(itr), ')']);
                
                        subplot(3,3,6)
                        imagesc(r_hat); colormap gray; axis image; colorbar;
                        title(['Reconstructed phase bias  (itr=',num2str(itr), ')']);
                
                        subplot(3,3,[7,8,9])
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
                    %}

                    if itr == max_itr %max_itrに達したとき更新終了
                        break;
                    end

                end %ミニバッチ終わり
    
                if itr == max_itr %max_itrに達したとき更新終了
                    break;
                end
            end %エポック終わり

            O_hat_results(:,:,trial) = O_hat;
            r_hat_results(:,:,trial) = r_hat;
            batch_es_results(:,trial) = batch_es;

        end %初期値（O_hat, r_hat）を降ってのループ終了
        
        %ここから品質評価
        batch_es_results_last = batch_es_results(max_itr,:);
        [min_batch_es_last,best_index] = min(batch_es_results_last);

        O_hat_best = O_hat_results(:,:,best_index);
        r_hat_best = r_hat_results(:,:,best_index);
        batch_es_best = batch_es_results(:,best_index);

        %サポート上のobjとO_hat_bestの相互相関
        O_hat_onSup = O_hat_best(row(1):row(end), col(1):col(end));
        obj_onSup = obj(row(1):row(end), col(1):col(end));
        O_hat_onSup_flip = rot90(O_hat_onSup, 2);
        corr_map = real(MyIFFT2(MyFFT2(abs(obj_onSup)) .* MyFFT2(abs(O_hat_onSup_flip))));
        
        %相互相関が最大となるindexを求め、サポート上のO_hat_bestのシフト量を求める
        [max_corr, max_corr_index] = max(corr_map(:));
        [max_corr_row, max_corr_col] = ind2sub(size(corr_map), max_corr_index);
        rows_shift = max_corr_row - ceil(length(corr_map)/2) ;
        cols_shift = max_corr_col - ceil(length(corr_map)/2) ;

        %24近傍シフト時の最良補正値の保存変数の初期化
        RMSE_o_best = 0;
        RMSE_o_abs_best = 0;
        RMSE_o_angle_best = 0;
        RMSE_r_best = 1000; 
        O_hat_corrected_best = zeros(N);
        exp_r_hat_corrected_best = exp(1i*ones(N));
        row_add_best = 0;
        col_add_best = 0;
        dif_bias_best = 0;

        %24近傍でRMSE_rが最小となるシフト量を探索
        for row_add = -2:2
            for col_add = -2:2
                %row_shift, col_shiftを24近傍にシフト
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
                
                %angle(O_hat_shifted)のオフセット量を推定し位相を補正
                dif_angle = wrapTo2Pi(angle(obj)) - wrapTo2Pi(angle(O_hat_shifted));
                %for [0,2pi]
                dif_angle_1 =  wrapTo2Pi(dif_angle);
                angle_offset_1 = sum(dif_angle_1(:))/nnz(sup);
                O_hat_corrected_1 = O_hat_shifted.*exp(1i*angle_offset_1);
                %for [-pi,pi]
                dif_angle_2 =  wrapToPi(dif_angle);
                angle_offset_2 = sum(dif_angle_2(:))/nnz(sup);
                O_hat_corrected_2 = O_hat_shifted.*exp(1i*angle_offset_2);

                %r_hatのオフセット量を推定し位相を補正
                dif_bias = r - r_hat_shifted;
                %for [0,2pi]
                dif_bias_1 =  wrapTo2Pi(dif_bias);
                bias_offset_1 = sum(dif_bias_1(:))/num_antenna;
                exp_r_hat_corrected_1 = exp(1i*(r_hat_shifted +bias_offset_1).*array);
                %for [-pi,pi]
                dif_bias_2 =  wrapToPi(dif_bias);
                bias_offset_2 = sum(dif_bias_2(:))/num_antenna;
                exp_r_hat_corrected_2 = exp(1i*(r_hat_shifted +bias_offset_2).*array);

                %RMSEの計算
                RMSE_o_1 = sqrt(mean(abs(O_hat_corrected_1(:) - obj(:)).^2));
                RMSE_o_2 = sqrt(mean(abs(O_hat_corrected_2(:) - obj(:)).^2));

                RMSE_o_abs = sqrt(mean((abs(O_hat_corrected_1(:)) - abs(obj(:))).^2));
                RMSE_o_angle_1 = sqrt(mean((wrapTo2Pi(angle(O_hat_corrected_1(:))) - wrapTo2Pi(angle(obj(:)))).^2));
                RMSE_o_angle_2 = sqrt(mean((wrapTo2Pi(angle(O_hat_corrected_2(:))) - wrapTo2Pi(angle(obj(:)))).^2));
                
                RMSE_r_1 = sqrt(sum(abs(exp_r_hat_corrected_1(:) - exp(1i*r(:))).^2)/num_antenna);
                RMSE_r_2 = sqrt(sum(abs(exp_r_hat_corrected_2(:) - exp(1i*r(:))).^2)/num_antenna);
                
                if RMSE_o_1 < RMSE_o_2
                    RMSE_o = RMSE_o_1;
                    RMSE_o_angle = RMSE_o_angle_1;
                    O_hat_corrected = O_hat_corrected_1;
                else
                    RMSE_o = RMSE_o_2;
                    RMSE_o_angle = RMSE_o_angle_2;
                    O_hat_corrected = O_hat_corrected_2;
                end

                if RMSE_r_1 < RMSE_r_2
                    RMSE_r = RMSE_r_1;
                    exp_r_hat_corrected = exp_r_hat_corrected_1;
                else
                    RMSE_r = RMSE_r_2;
                    exp_r_hat_corrected = exp_r_hat_corrected_2;
                end

                %RMSE_rが最小の時の各補正値を保持
                if RMSE_r < RMSE_r_best
                    RMSE_o_best = RMSE_o;
                    RMSE_o_abs_best =  RMSE_o_abs;
                    RMSE_o_angle_best = RMSE_o_angle;
                    RMSE_r_best = RMSE_r;
                    O_hat_corrected_best = O_hat_corrected;
                    exp_r_hat_corrected_best = exp_r_hat_corrected;
                    row_add_best = row_add;
                    col_add_best = col_add;
                    dif_angle_best = dif_angle;
                    dif_bias_best = dif_bias;
                end
            end
        end

        % 結果の表示
        figure(30);
        subplot(2,2,1);imagesc(wrapTo2Pi(dif_angle_best)); colormap gray; axis image; colorbar; title('dif angle[0~2pi]');
        subplot(2,2,2);imagesc(wrapToPi(dif_angle_best)); colormap gray; axis image; colorbar; title('dif angle[-pi~pi]');
        subplot(2,2,3);imagesc(wrapTo2Pi(dif_bias_best)); colormap gray; axis image; colorbar; title('dif bias[0~2pi]');
        subplot(2,2,4);imagesc(wrapToPi(dif_bias_best)); colormap gray; axis image; colorbar; title('dif bias[-pi~pi]');
        drawnow();

        figure(100);
        gcf.Position = [714 91 818 775];
        subplot(4,3,1)
        imagesc(abs(obj)); colormap gray; axis image; colorbar;
        title('Original amplitude');

        subplot(4,3,2)
        imagesc(wrapTo2Pi(angle(obj))); colormap gray; axis image; colorbar; clim([0,2*pi]);
        title('Original phase');
        
        subplot(4,3,3)
        imagesc(r); colormap gray; axis image; colorbar; clim([0, 2*pi]);
        title('Original phase bias');
        
        subplot(4,3,4)
        imagesc(abs(O_hat_best)); colormap gray; axis image; colorbar;
        title('Reconstructed amplitude');

        subplot(4,3,5)
        imagesc(wrapTo2Pi(angle(O_hat_best))); colormap gray; axis image; colorbar; clim([0,2*pi]);
        title('Reconstructed phase');
        
        subplot(4,3,6)
        imagesc(r_hat_best); colormap gray; axis image; colorbar; clim([0, 2*pi]);
        title('Reconstructed phase bias');
        
        subplot(4,3,7)
        imagesc(abs(O_hat_corrected_best)); colormap gray; axis image; colorbar;
        title(['RMSE_{amp}=',num2str(RMSE_o_abs_best, 4),',RMSE_{total}=',num2str(RMSE_o_best, 4)]);

        subplot(4,3,8)
        imagesc(wrapTo2Pi(angle(O_hat_corrected_best))); colormap gray; axis image; colorbar; clim([0,2*pi]);
        title(['RMSE_{phase}=',num2str(RMSE_o_angle_best, 4),',RMSE_{total}=',num2str(RMSE_o_best, 4)]);
        
        subplot(4,3,9)
        imagesc(wrapTo2Pi(angle(exp_r_hat_corrected_best))); colormap gray; axis image; colorbar; clim([0, 2*pi]);
        title(['RMSE=',num2str(RMSE_r_best, 4)]);
        
        subplot(4,3,10)
        imagesc(corr_map); colormap gray; axis image; colorbar;
        title('Correlation map');
        
        subplot(4,3,[11,12])
        semilogy(batch_es_best(1:max_itr));
        title('|S_{hat} - S|^2');

        drawnow();

        %結果を保存
        save_dir = sprintf('./figure_M1_検証/GaussianSigma_0112/M1_%s_sigma%.1f_%s_N%d_K%d_sup%d_noise%d/',array_name,sigma,obj_name,N,K,sup_size,noiseLv);
        mkdir(save_dir);
        filename_fig = sprintf('%s%d.fig',save_dir,seed);
        filename_png = sprintf('%s%d.png',save_dir,seed);
       
        %ファイルを保存
        savefig(filename_fig);
        print(filename_png, '-dpng', '-r300');

        finishMessage = sprintf('sigma=%.1f (%d/%d),Gaussアレイseed(%d/%d)の結果を保存, RMSE:o_abs=%.4f,o_angle=%.4f,o_total=%.4f,r=%.4f, [row,col]_add=[%d,%d])',sigma,idx_sigma,sigmas_len,seed,num_Gauss_trial,RMSE_o_abs_best,RMSE_o_angle_best,RMSE_o_best,RMSE_r_best,row_add_best,col_add_best);
        disp(finishMessage);

        %RMSEを保存
        RMSE_tmp_o(seed) = RMSE_o_best;
        RMSE_tmp_o_abs(seed) = RMSE_o_abs_best;
        RMSE_tmp_o_angle(seed) = RMSE_o_angle_best;
        RMSE_tmp_r(seed) = RMSE_r_best;

        clearvars phi A      
    end %アンテナ配置seed（num_Gauss_trial通り）を切り替えてのループ終了
    
    RMSEs_o(idx_sigma) = mean(RMSE_tmp_o);
    stds_rmse_o(idx_sigma) = std(RMSE_tmp_o);
    RMSEs_o_abs(idx_sigma) = mean(RMSE_tmp_o_abs);
    stds_rmse_o_abs(idx_sigma) = std(RMSE_tmp_o_abs);
    RMSEs_o_angle(idx_sigma) = mean(RMSE_tmp_o_angle);
    stds_rmse_o_angle(idx_sigma) = std(RMSE_tmp_o_angle);
    RMSEs_r(idx_sigma) = mean(RMSE_tmp_r);
    stds_r(idx_sigma) = std(RMSE_tmp_r);

end

RMSE_path = './figure_M1_検証/GaussianSigma_0112/RMSE/';
mkdir(RMSE_path);

figure(1000);
subplot(1,3,1)
e_abs = errorbar(sigmas,RMSEs_o_abs,stds_rmse_o_abs,'LineWidth',1);
hold on;
e_angle = errorbar(sigmas,RMSEs_o_angle,stds_rmse_o_angle,'LineWidth',1,'LineStyle','--');
hold off;
title('RMSE of object (amplitude,Phase)');
xlabel('Sigma');
ylabel('Reconstruction RMSE');
legend([e_abs,e_angle],{'Amplitude','Phase'})

subplot(1,3,2)
errorbar(sigmas,RMSEs_o,stds_rmse_o,'LineWidth',1);
title('RMSE of object (total)');
xlabel('Sigma');
ylabel('Reconstruction RMSE');

subplot(1,3,3)
errorbar(sigmas,RMSEs_r,stds_r,'LineWidth',1);
title('RMSE of phase bias');
xlabel('Sigma');
ylabel('Reconstruction RMSE');

%RMSEのグラフを保存
RMSE_fig = sprintf('%sM1_%s_%s_N%d_sup%d_noise%d_numE%d.fig',RMSE_path,obj_name,array_name,N,sup_size,noiseLv,num_Gauss_trial);
RMSE_png = sprintf('%sM1_%s_%s_N%d_sup%d_noise%d_numE%d.png',RMSE_path,obj_name,array_name,N,sup_size,noiseLv,num_Gauss_trial);
savefig(RMSE_fig);
print(RMSE_png, '-dpng', '-r300');

clearvars matrix img img_gray 






