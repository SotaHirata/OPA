close all;
clear all;clc;

%乱数seedの固定
rng(0);

% GPU
GPU_num = 4;
gpuDevice(GPU_num);
reset(gpuDevice(GPU_num));
executionEnvironment = 'gpu';
gpurng(0);

%パラメタ
M = 10;     %uniformアレイの1辺の長さ
N = 127;    %アンテナ数
K = N^2*4;    %照射パターン数

%SGDの設定
num_epoch = 60;  %エポック数
batch_size = 2^8; %バッチサイズ
num_itr = (ceil(K/batch_size))*num_epoch; %反復回数
data_indice = randperm(K); %batch列を用意

%強度分布画像を生成（N×N）
%obj = gpuArray(double(MyRect(N,[N/2,N/7],[N/2,N/3]) +MyRect(N,[N/2,N/7],[N/2,2*N/3]))) 

img = imread('peppers.png');
img_resized = imresize(img, [N, N]);
img_gray = double(rgb2gray(img_resized)) ;
obj = img_gray / max(img_gray(:));
obj = gpuArray(double(obj.*MyRect(N, N/1.5)));


%サポート
sup =gpuArray(double(MyRect(N, N/1.5)));

%アンテナ位置を表す行列（N×N）
%array = gpuArray(double(MyRect(N, M))); %for uniformアレイ
%load('random_array_9.mat') ;
%array = gpuArray(double(randomarray));
load('Costasarray_N127.mat') ;
array = matrix;
%array = gpuArray(double(matrix));%for Costasアレイ

%位相シフトKパターン（N×N×K）
phi = array.*rand(N,N,K)*2*pi;
%phi = array.*rand(N,N,K,'double','gpuArray')*2*pi;

%アンテナ配置×位相シフト（N×N×K）
A = array.*exp(1i*phi);
%A = array.*gpuArray(double(exp(1i*phi))); 

%位相バイアス（N×N）
r = array.*rand(N,N,'double','gpuArray')*2*pi; 

%PDの観測強度（K×1配列）
S = zeros(1,K,'double','gpuArray');
for batch_start = 1:batch_size:K
    %照射パターン
    batch_F = MyFFT2(A(:,:,batch_start:min(batch_start+batch_size-1, K)).*gpuArray(double(exp(1i*r))));
    S(batch_start:min(batch_start+batch_size -1, K)) = sum(abs(batch_F).^2.*obj.*sup, [1,2]);
end
S = reshape(S, [K,1]); 

%ここから逆問題
figure(100);
O_hat = ones(N,'double','gpuArray'); %Oの初期値（N×N）
r_hat = array.*rand(N,'double','gpuArray')*2*pi; %rの初期値（N×N）
batch_es = zeros(num_itr,1,'double','gpuArray');

%adamの初期パラメタ
m_O = zeros(N,'double','gpuArray');
v_O = zeros(N,'double','gpuArray');
alpha_O = 1e-1;
beta_1_O = 0.99;
beta_2_O = 0.999;
epsilon_O = 1e-8;

m_r = zeros(N,'double','gpuArray');
v_r = zeros(N,'double','gpuArray');
alpha_r = 1e-1;
beta_1_r = 0.99;
beta_2_r = 0.999;
epsilon_r = 1e-8;

%alpha = 1e-8; %Oの更新幅
%beta = 1e-6; %rの更新幅

%TVの初期parameter
rho_O = 0; %TVなし
%rho_O = 1e-3; %TVあり
tv_th = 1e-2;
tv_tau = 0.05;
tv_iter = 4; %TVの反復数

v_TV_O =  ones(N,'double','gpuArray');
u_TV_O = zeros(N,'double','gpuArray');

%0<O<1 constraint
mu1 = 1e8; %for non-negative constraint
mu2 = 1e8; %for O<=1 constraint

elapsed_times = zeros(floor(num_itr/100), 1);
itr = 0;
hundreds = 0;

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

        %non-negative constraint
        ReLU1_O = -O_hat;
        ReLU1_O(O_hat>0) = 0;
        dReLU1_O = zeros(N, 'double','gpuArray');
        dReLU1_O(O_hat<0) = -1;

        %O<=1 constraint
        ReLU2_O = O_hat - 1;
        ReLU2_O(O_hat<1) = 0;
        dReLU2_O = zeros(N, 'double','gpuArray');
        dReLU2_O(O_hat>=1) = 1;

        %O,rの勾配
        st_O = 2*sum(abs(batch_F).^2.*reshape(batch_e, [1,1,length(batch_idx)]), 3) + 2.*rho_O.*(O_hat - (v_TV_O - u_TV_O)) + 2*mu1*dReLU1_O.*ReLU1_O + 2*mu2*dReLU2_O.*ReLU2_O;
        st_r = 2*(-1i*exp(-1i*r_hat)).*sum(2*conj(batch_A).*MyIFFT2(batch_F.*O_hat.*reshape(batch_e, [1,1,length(batch_idx)])),3);
            
        %Adam
        [st_O, m_O, v_O] = Adam_func(st_O,m_O,v_O,itr,alpha_O,beta_1_O,beta_2_O,epsilon_O);
        [st_r, m_r, v_r] = Adam_func(st_r,m_r,v_r,itr,alpha_r,beta_1_r,beta_2_r,epsilon_r);
       
        %O,rの更新
        O_hat = (O_hat - st_O).*sup; %Oの更新式 
        r_hat = r_hat - real(st_r); %rの更新式
    
        %v,uの更新
        v_TV_O = reshape(MyTVpsi_ND(O_hat + u_TV_O, tv_th, tv_tau, tv_iter, [N, N]), [N, N]);
        u_TV_O = u_TV_O + (O_hat - v_TV_O);
    
        if rem(itr, 100)==0    %描画
            hundreds = hundreds + 1;
    
            subplot(3,2,1)
            imagesc(real(O_hat)); colormap gray; axis image; colorbar;
            title(['Reconstructed image ( itr=',num2str(itr), ' )']);
    
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
end




