close all;
clear all;clc;

%乱数seedの固定
rng(0);

%パラメタ
M = 6;     %uniformアレイの1辺の長さ
N = M^2;    %アンテナ数
K =  M^4*4;    %照射パターン数
maskD = 1000; %PDの受光範囲の直径

%複素振幅画像を生成（N×N）
img = imread('peppers.png');
img_resized = imresize(img, [N, N]);
img_gray = double(rgb2gray(img_resized)) ;
obj = img_gray / max(img_gray(:));
obj = obj.*exp(1i*2*pi*rot90(obj)); %適切な位相を付与

%アンテナ位置を表す行列（N×N）
array = MyRect(N, M); %for uniformアレイ
%load('Costasarray_N127.mat') ;
%array = matrix;%for Costasアレイ

%ランダムな固定された初期位相（0〜2pi）
r = array.*rand(N,N)*2*pi; 

%ランダムな位相シフト（0〜2pi）Kパターン
phi = array.*rand(N,N,K)*2*pi; 

%PDの受光範囲マスクを作成
mask = MyCirc(N, maskD); 

%PDの観測強度（K×1配列）
%S = reshape(sum(abs(MyIFFT2(MyFFT2(exp(1i*(phi+r)).*array).*obj)).^2.*mask, [1,2]), [K,1]); 

O = MyRect(N, N/2);
H = rand(K,N^2).*exp(1i*2*pi*rand(K,N^2));
S = abs(H*O(:)).^2;

%ここから逆問題
figure(100);
O_hat = ones(N^2,1); %Oの初期値（N^2×1）
alpha = 1e-7; %Oの更新幅
num_itr = 10000; %反復回数
itr=1; %itrカウンタ

%H = reshape(permute(MyFFT2(exp(1i*(phi+r)).*array), [3,1,2]), [K, N^2]); 
%H_H = H';

es = zeros(num_itr,1);

while itr <= num_itr
    %Oの更新
    S_hat = H*O_hat;
    e = abs(S_hat).^2 - S;
    es(itr) = mean(abs(e).^2, 'all');

    grad_O = H'*(S_hat.*e);
    O_hat = O_hat - alpha*grad_O; %Oの更新式

    if rem(itr, 100)==0    %描画
        O_hat_2D = reshape(O_hat, [N, N]);

        subplot(2,2,1)
        imagesc(abs(O_hat_2D)); colormap gray; axis image; colorbar;
        title(['Reconstructed amplitude ( itr=',num2str(itr), ' )']);

        subplot(2,2,2)
        imagesc(angle(O_hat_2D)); colormap gray; axis image; colorbar;
        title(['Reconstructed phase  ( itr=',num2str(itr), ' )']);

        subplot(2,2,[3,4])
        semilogy(es(1:itr));

        drawnow();
    end
    itr = itr+1;
end


