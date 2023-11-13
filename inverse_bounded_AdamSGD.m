close all;
clear all;clc;

%乱数seedの固定
rng(0);

%パラメタ
M = 4;     %uniformアレイの1辺の長さ
N = M^2;    %アンテナ数
K =  N^2*4;    %照射パターン数
maskD = N-4; %PDの受光範囲の直径

%複素振幅画像を生成（N×N）
obj = MyRect(N, N/3);
%obj = MyRect(N, [N/2,N/4],[N/2, N/4]) + MyRect(N, [N/2,N/4],[N/2,3*N/4]);

%サポート
sup = MyRect(N, N-6);

%アンテナ位置を表す行列（N×N）
%array = MyRect(N, M); %for uniformアレイ
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

%PDの受光範囲マスクを作成（N×N）
mask = MyCirc(N, maskD); 

%PDの観測強度（K×1配列）
S = reshape(sum(abs(MyIFFT2(MyFFT2(A.*exp(1i*r)).*obj.*sup)).^2.*mask, [1,2]), [K,1]); 


%ここから逆問題
figure(100);
O_hat = ones(N); %Oの初期値（N×N）
r_hat = array.*rand(N)*2*pi; %rの初期値（N×N）
num_itr = 5000; %反復回数
batch_size = 2^6; %バッチサイズ
%es = zeros(num_itr,1);
batch_es = zeros(num_itr,1);

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

%alpha = 1e-6; %Oの更新幅
%beta = 1e-4; %rの更新幅

hundreds = 0;
elapsed_times = zeros(num_itr/100, 1);

tic;
for itr = 1:num_itr

    random_idx = randperm(K, batch_size);
    batch_A = A(:,:,random_idx);
    batch_F = MyFFT2(batch_A.*exp(1i*r_hat));
    batch_I = MyIFFT2(batch_F.*O_hat); 
    batch_S_hat = reshape(sum(abs(batch_I).^2.*mask, [1,2]), [batch_size,1]); 
    batch_e = batch_S_hat - S(random_idx);
    batch_es(itr) = mean(abs(batch_e).^2, 'all');

    %全体の損失関数の確認用
    %F = MyFFT2(A.*exp(1i*r_hat));
    %I = MyIFFT2(F.*O_hat); 
    %S_hat = reshape(sum(abs(I).^2.*mask, [1,2]), [K,1]); 
    %e = S_hat- S;
    %es(itr) = mean(abs(e).^2, 'all');
    
    %O,rの勾配
    st_O = 2*sum(2*conj(batch_F).*MyFFT2(batch_I.*mask.*reshape(batch_e,[1,1,batch_size])),3);
    st_r = 2*(-1i*exp(-1i*r_hat)).*sum(2*conj(batch_A).*MyIFFT2(conj(O_hat).*MyFFT2(batch_I.*mask.*reshape(batch_e,[1,1,batch_size]))),3);
    
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
        semilogy(batch_es(1:itr));
        title(['|S_{hat} - S|^2 (', num2str(sum(elapsed_times)/hundreds,4),'sec/100itr) ','(batchsize=',num2str(batch_size),')']);
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


