close all;
clear all;clc;

%乱数seedの固定
rng(0);

%パラメタ
M = 4;     %uniformアレイの1辺の長さ
N = M^2;    %アンテナ数
K = N^2*4;    %照射パターン数

%強度分布画像を生成（N×N）
obj = MyRect(N, N/2);

%アンテナ位置を表す行列（N×N）
array = MyRect(N, M); %for uniformアレイ
%array = zeros(N);
%indices = randperm(N^2, N); % N個のランダムなインデックスを生成
%array(indices) = 1;
%load('Costasarray_N16.mat') ;
%array = matrix;%for Costasアレイ



%位相シフトKパターン（N×N×K）
phi = array.*rand(N,N,K)*2*pi;

%アンテナ配置×位相シフト（N×N×K）
A = exp(1i*phi).*array; 

%位相バイアス（N×N）
r = array.*ones(N,N)*2*pi; %for uniformアレイ

%照射パターン（N×N×K）
F = MyFFT2(A.*exp(1i*r));

%{
figure(1);
subplot(2,2,1)
imagesc(abs(F(:,:,5)).^2); colormap gray; axis image; colorbar;
drawnow;
%}

%PDの観測強度（K×1配列）
S = reshape(sum(abs(F).^2.*obj, [1,2]), [K,1]); 

%ここから逆問題
figure(100);
O_hat = ones(N); %Oの初期値（N×N）
r_hat = array.*rand(N)*2*pi; %rの初期値（N×N）
alpha = 1e-8; %Oの更新幅
beta = 1e-6; %rの更新幅
num_itr = 400; %反復回数
itr=1; %itrカウンタ

es = zeros(num_itr,1);

while itr <= num_itr

    F = MyFFT2(A.*exp(1i*r_hat));
    S_hat = reshape(sum(abs(F).^2.*O_hat, [1,2]), [K,1]);

    e = S_hat- S;
    es(itr) = mean(abs(e).^2, 'all');

    grad_O = 2*sum(abs(F).^2.*reshape(e, [1,1,K]), 3);
    grad_r = 2*(-1i*exp(-1i*r_hat)).*sum(2*conj(A).*MyIFFT2(F.*O_hat.*reshape(e, [1,1,K])),3);
    
    O_hat = O_hat - alpha*grad_O; %Oの更新式
    r_hat = r_hat - beta*real(grad_r); %rの更新式

    if rem(itr, 100)==0    %描画
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
        title('|S_{hat} - S|^2');

        drawnow(); 
    end

    itr = itr+1;
end


