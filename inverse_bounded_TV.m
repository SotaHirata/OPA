close all;
clear all;clc;

%乱数seedの固定
rng(0);

%パラメタ
M = 4;     %uniformアレイの1辺の長さ
N = M^2;    %アンテナ数
K =  N^2*4;    %照射パターン数
maskD = N/2; %PDの受光範囲の直径

%複素振幅画像を生成（N×N）
obj = MyRect(N, N);

%サポート
sup = MyRect(N, N);

%アンテナ位置を表す行列（N×N）
%array = MyRect(N, M); %for uniformアレイ
%load('random_array_9');
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
num_itr = 6000; %反復回数
es = zeros(num_itr,1);

%adamの初期パラメタ
m_O = zeros(N);
v_O = zeros(N);
alpha_O = 1e-1;
beta_1_O = 0.97;
beta_2_O = 0.999;
epsilon_O = 1e-8;

m_r = zeros(N);
v_r = zeros(N);
alpha_r = 1e-1;
beta_1_r = 0.97;
beta_2_r = 0.999;
epsilon_r = 1e-8;

%alpha = 1e-6; %Oの更新幅
%beta = 1e-4; %rの更新幅

%TVの初期パラメタ
%rho_O = 0;
rho_O = 6e2; 
tv_th = 1e-2;
tv_tau = 0.05;
tv_iter = 4; %TVの反復数

v_TV_O =  ones(N);
u_TV_O = zeros(N);

%経過時間計算用
hundreds = 0;
elapsed_times = zeros(num_itr/100, 1);

tic;
for itr = 1:num_itr

    F = MyFFT2(A.*exp(1i*r_hat));
    I = MyIFFT2(F.*O_hat); 
    S_hat = reshape(sum(abs(I).^2.*mask, [1,2]), [K,1]); 

    e = S_hat- S;
    es(itr) = mean(abs(e).^2, 'all');
    
    %O,rの勾配
    st_O = 2*sum(2*conj(F).*MyFFT2(I.*mask.*reshape(e,[1,1,K])),3) + 2.*rho_O.*(O_hat - (v_TV_O - u_TV_O));
    st_r = 2*(-1i*exp(-1i*r_hat)).*sum(2*conj(A).*MyIFFT2(conj(O_hat).*MyFFT2(I.*mask.*reshape(e,[1,1,K]))),3);
    
    %Adam
    [st_O, m_O, v_O] = Adam_func(st_O,m_O,v_O,itr,alpha_O,beta_1_O,beta_2_O,epsilon_O);
    [st_r, m_r, v_r] = Adam_func(st_r,m_r,v_r,itr,alpha_r,beta_1_r,beta_2_r,epsilon_r);
    
    %O,rの更新
    O_hat = (O_hat - st_O).*sup; %Oの更新式
    r_hat = r_hat - real(st_r) ; %rの更新式

    %v,uの更新
    v_TV_O_re = reshape(MyTVpsi_ND(real(O_hat + u_TV_O), tv_th, tv_tau, tv_iter, [N, N]), [N, N]);
    v_TV_O_im = reshape(MyTVpsi_ND(imag(O_hat + u_TV_O), tv_th, tv_tau, tv_iter, [N, N]), [N, N]);
    v_TV_O = v_TV_O_re + 1i .*v_TV_O_im;
    u_TV_O = u_TV_O + (O_hat - v_TV_O);

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


