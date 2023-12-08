close all;
clear all;clc;

%乱数seedの固定
rng(0);

%パラメタ
M = 4;     %uniformアレイの1辺の長さ
N = M^2;    %アンテナ数
K =  N^2*4;    %照射パターン数
maskD = N/1.5; %PDの受光範囲の直径

%複素振幅画像を生成（N×N）
obj = MyRect(N, N/3).*exp(1i*0.5*pi);
%obj = MyRect(N, [N/2,N/4],[N/2, N/4]) + MyRect(N, [N/2,N/4],[N/2,3*N/4]);

%サポート
sup = MyRect(N, N/2);
%sup = MyRect(N, N);

%アンテナ位置を表す行列（N×N）
array = MyRect(N, M); %for uniformアレイ
%load('random_array_0.mat') ;
%array = randomarray;
load('Costasarray_N16.mat') ;
array = matrix;%for Costasアレイ


%位相シフトKパターン（N×N×K）
phi = array.*rand(N,N,K)*2*pi; 

%アンテナ配置×位相シフト（N×N×K）
A = exp(1i*phi).*array; 

%位相バイアス（N×N）
r = array.*rand(N)*2*pi; 

%アンテナ1個の遠視野
%U = MyGaussian2D(N,N,N/2+1,N/2+1,N/8,N/8);
%U = U/max(U(:));
U = ones(N);

%PDの受光範囲マスクを作成（N×N）
mask = MyCirc(N, maskD); 

%PDの観測強度（K×1配列）
S = reshape(sum(abs(MyIFFT2(MyFFT2(A.*exp(1i*r)).*U.*obj.*sup)).^2.*mask, [1,2]), [K,1]); 


%ここから逆問題
figure(100);
O_hat = ones(N); %Oの初期値（N×N）
r_hat = array.*rand(N)*2*pi; %rの初期値（N×N）
num_itr = 4000; %反復回数
es = zeros(num_itr,1);

%adamの初期パラメタ
m_O = zeros(N);
v_O = zeros(N);
alpha_O = 5e-2;
beta_1_O = 0.93;
beta_2_O = 0.999;
epsilon_O = 1e-8;

m_r = zeros(N);
v_r = zeros(N);
alpha_r = 5e-2;
beta_1_r = 0.93;
beta_2_r = 0.999;
epsilon_r = 1e-8;

hundreds = 0;
elapsed_times = zeros(num_itr/100, 1);

%alpha = 1e-6; %Oの更新幅
%beta = 1e-4; %rの更新幅

tic;
for itr = 1:num_itr

    F = MyFFT2(A.*exp(1i*r_hat)).*U;
    I = MyIFFT2(F.*O_hat); 
    S_hat = reshape(sum(abs(I).^2.*mask, [1,2]), [K,1]); 

    e = S_hat- S;
    es(itr) = mean(abs(e).^2, 'all');
    
    %O,rの勾配
    st_O = 2*sum(2*conj(F).*MyFFT2(I.*mask.*reshape(e,[1,1,K])),3);
    st_r = 2*(-1i*exp(-1i*r_hat)).*sum(2*conj(A).*MyIFFT2(conj(O_hat.*U).*MyFFT2(I.*mask.*reshape(e,[1,1,K]))),3);
    
    %Adam
    [st_O, m_O, v_O] = Adam_func(st_O,m_O,v_O,itr,alpha_O,beta_1_O,beta_2_O,epsilon_O);
    [st_r, m_r, v_r] = Adam_func(st_r,m_r,v_r,itr,alpha_r,beta_1_r,beta_2_r,epsilon_r);
    
    %O,rの更新
    O_hat = (O_hat - st_O).*sup; %Oの更新式
    r_hat = r_hat - real(st_r) ; %rの更新式

    if rem(itr, 100)==0    %描画
        hundreds = hundreds + 1;

        subplot(3,3,1)
        imagesc(abs(O_hat)); colormap gray; axis image; colorbar;
        title(['Reconstructed amplitude ( itr=',num2str(itr), ' )']);

        subplot(3,3,2)
        imagesc(angle(O_hat)); colormap gray; axis image; colorbar;
        title(['Reconstructed phase  ( itr=',num2str(itr), ' )']);

        subplot(3,3,3)
        imagesc(r_hat); colormap gray; axis image; colorbar;
        title(['Reconstructed phase bias ( itr=',num2str(itr), ' )']);

        subplot(3,3,4)
        imagesc(abs(obj)); colormap gray; axis image; colorbar;
        title(['Original object amplitude ( itr=',num2str(itr), ' )']);

        subplot(3,3,5)
        imagesc(angle(obj)); colormap gray; axis image; colorbar;
        title(['Original object phase  ( itr=',num2str(itr), ' )']);

        subplot(3,3,6)
        imagesc(r); colormap gray; axis image; colorbar;
        title(['Original phase bias ( itr=',num2str(itr), ' )']);

        subplot(3,3,[7,8,9])
        semilogy(es(1:itr));
        title(['|S_{hat} - S|^2  (', num2str(sum(elapsed_times)/hundreds,4),'sec/100itr)']);

        drawnow();
        
        %イテレーションごとの所要時間
        elapsed_time = toc;
        fprintf('イテレーション %d の経過時間: %f 秒\n', itr, elapsed_time);
        elapsed_times(hundreds) = elapsed_time;
        if itr ~=num_itr
           tic;
        end 
    end
end

%abs(obj)とabs(O_hat)の相互相関を計算
O_hat_flip = rot90(abs(O_hat), 2);
corr_map = MyIFFT2(MyFFT2(abs(obj)) .* MyFFT2(O_hat_flip));

%相互相関が最大となるindexを求め、O_hatのシフト量を求める
[max_corr, max_corr_index] = max(corr_map(:));
[max_corr_row, max_corr_col] = ind2sub(size(corr_map), max_corr_index);
rows_shift = max_corr_row - floor(N/2);
cols_shift = max_corr_col - floor(N/2);

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

%angle(O_hat)の定数加算量を推定し補正
dif_angle = angle(obj) - wrapTo2Pi(angle(O_hat_shifted.*abs(obj)));
angle_shift = sum(dif_angle(:))/nnz(abs(obj));
O_hat_shifted = O_hat_shifted.*exp(1i*angle_shift);

%r_hatの定数加算量を推定し位相を補正
dif_bias = wrapTo2Pi(r - r_hat_flattened);
bias_shift = sum(dif_bias(:))/N;
r_hat_flattened = wrapTo2Pi((r_hat_flattened() + bias_shift).*array);

%RMSEの計算
RMSE_o = sqrt(mean(abs(O_hat_shifted(:) - obj(:)).^2));
RMSE_r = sqrt(sum((r_hat_flattened(:) - r(:)).^2)/N);
%peaksnr = psnr(abs(O_hat),abs(obj));

% 結果の表示
figure('Position',[714 91 818 775]);
subplot(4,3,1)
imagesc(abs(obj)); colormap gray; axis image; colorbar;
title('Original object amplitude');

subplot(4,3,2)
imagesc(angle(obj)); colormap gray; axis image; colorbar; clim([0, 2*pi]);
title('Original object phase');

subplot(4,3,3)
imagesc(r); colormap gray; axis image; colorbar; clim([0, 2*pi]);
title('Original phase bias');

subplot(4,3,4)
imagesc(abs(O_hat)); colormap gray; axis image; colorbar;
title('Reconstructed amplitude');

subplot(4,3,5)
imagesc(angle(O_hat)); colormap gray; axis image; colorbar; %clim([0, 2*pi]);
title('Reconstructed phase');

subplot(4,3,6)
imagesc(r_hat); colormap gray; axis image; colorbar; clim([0, 2*pi]);
title('Reconstructed phase bias');

subplot(4,3,7)
imagesc(abs(O_hat_shifted)); colormap gray; axis image; colorbar;
title('Compensated amplitude');

subplot(4,3,8)
imagesc(wrapTo2Pi(angle(O_hat_shifted))); colormap gray; axis image; colorbar; clim([0, 2*pi]);
title('Compensated phase');

subplot(4,3,9)
imagesc(r_hat_flattened); colormap gray; axis image; colorbar; clim([0, 2*pi]);
title('Compensated phase bias');

subplot(4,3,10)
imagesc(corr_map); colormap gray; axis image; colorbar;
title('Correlation map');

subplot(4,3,[11,12])
semilogy(es(1:itr));
title(['|S_{hat} - S|^2  (', num2str(sum(elapsed_times)/hundreds,4),'sec/100itr)']);

% テキストボックスを追加
textbox = uicontrol('style', 'text', 'String', ['RMSE (object) = ',num2str(RMSE_o, 4)] , ...
    'Position', [150, 30, 200, 20], 'Callback', @textboxCallback,'FontSize', 15);

textbox = uicontrol('style', 'text', 'String', ['RMSE (phase bias) = ',num2str(RMSE_r, 4)] , ...
    'Position', [400, 30, 300, 20], 'Callback', @textboxCallback,'FontSize', 15);

drawnow();

% コールバック関数
function textboxCallback(src, event)
    % テキストボックスの中身を取得
    textValue = get(src, 'String');  
    % 何かしらの処理（この例では表示のみ）
    disp(['RMSE (object) = ', textValue]);
end

