close all;
clear all;clc;

%乱数seedの固定
rng(0);

%パラメタ
M = 4;     %uniformアレイの1辺の長さ
N = M^2;    %アンテナ数
K = N^2*4;    %照射パターン数

%強度分布画像を生成（N×N）
obj = MyRect(N, N/5, [N/2-2, N/2-2]);

%サポート
sup = MyRect(N, N);

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
r = array.*rand(N,N)*2*pi; %for uniformアレイ

%アンテナ1個の遠視野
U = MyGaussian2D(N,N,N/2+1,N/2+1,N/8,N/8);
U = U/max(U(:));
%U = ones(N);

%照射パターン（N×N×K）
F = MyFFT2(A.*exp(1i*r)).*U;

%PDの観測強度（K×1配列）
S = reshape(sum(abs(F).^2.*obj.*sup, [1,2]), [K,1]); 

%ここから逆問題
figure(100);
O_hat = ones(N); %Oの初期値（N×N）
r_hat = array.*rand(N)*2*pi; %rの初期値（N×N）
num_itr = 5000; %反復回数
es = zeros(num_itr,1);

%adamの初期パラメタ
m_O = zeros(N);
v_O = zeros(N);
alpha_O = 5e-2;
beta_1_O = 0.9;
beta_2_O = 0.999;
epsilon_O = 1e-8;

m_r = zeros(N);
v_r = zeros(N);
alpha_r = 5e-2;
beta_1_r = 0.9;
beta_2_r = 0.999;
epsilon_r = 1e-8;

%alpha = 1e-8; %Oの更新幅
%beta = 1e-6; %rの更新幅

hundreds = 0;
elapsed_times = zeros(num_itr/100, 1);

tic;
for itr = 1:num_itr

    F = MyFFT2(A.*exp(1i*r_hat)).*U;
    S_hat = reshape(sum(abs(F).^2.*O_hat, [1,2]), [K,1]);

    e = S_hat- S;
    es(itr) = mean(abs(e).^2, 'all');

    %O,rの勾配
    st_O = 2*sum(abs(F).^2.*reshape(e, [1,1,K]), 3);
    st_r = 2*(-1i*exp(-1i*r_hat)).*sum(2*conj(A).*MyIFFT2(U.*F.*O_hat.*reshape(e, [1,1,K])),3);
        
    %Adam
    [st_O, m_O, v_O] = Adam_func(st_O,m_O,v_O,itr,alpha_O,beta_1_O,beta_2_O,epsilon_O);
    [st_r, m_r, v_r] = Adam_func(st_r,m_r,v_r,itr,alpha_r,beta_1_r,beta_2_r,epsilon_r);
   
    %O,rの更新
    O_hat = (O_hat - st_O).*sup; %Oの更新式
    r_hat = r_hat - real(st_r); %rの更新式

    if rem(itr, 100)==0    %描画
        hundreds = hundreds + 1;
        
        subplot(3,2,1)
        imagesc(O_hat); colormap gray; axis image; colorbar;
        title(['Reconstructed object ( itr=',num2str(itr), ' )']);

        subplot(3,2,2)
        imagesc(r_hat); colormap gray; axis image; colorbar;
        title(['Reconstructed phase bias  ( itr=',num2str(itr), ' )']);
        
        subplot(3,2,3)
        imagesc(obj); colormap gray; axis image; colorbar;
        title(['Original object ( itr=',num2str(itr), ' )']);

        subplot(3,2,4)
        imagesc(r); colormap gray; axis image; colorbar;
        title(['Original phase bias  ( itr=',num2str(itr), ' )']);

        subplot(3,2,[5,6])
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



