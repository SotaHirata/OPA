close all;
clear all;clc;

%乱数seedの固定
rng(0);

%パラメタ
M = 4;     %uniformアレイの1辺の長さ
N = M^2;    %アンテナ数
K =  N^2*4;    %照射パターン数
maskD = N/4; %PDの受光範囲の直径

%複素振幅画像を生成（N×N）
obj = MyRect(N, N/2);

%アンテナ位置を表す行列（N×N）
array = MyRect(N, M); %for uniformアレイ

%ランダムな固定された初期位相（0〜2pi）
r = array.*rand(N,N)*2*pi; 

%ランダムな位相シフト（0〜2pi）Kパターン
phi = array.*rand(N,N,K)*2*pi; 

%PDの受光範囲マスクを作成
mask = MyCirc(N, maskD); 

%PDの観測強度（K×1配列）
S = reshape(sum(abs(MyIFFT2(MyFFT2(exp(1i*(phi+r)).*array).*obj)).^2.*mask, [1,2]), [K,1]); 


%ここから逆問題
figure(100);
O_hat = ones(N); %Oの初期値（N×N）
alpha = 1e-7; %Oの更新幅
num_itr = 10000; %反復回数
itr=1; %itrカウンタ

es = zeros(num_itr,1);

while itr <= num_itr
    %Oの更新
    F = MyFFT2(array.*exp(1i*(phi+r)));
    I = MyIFFT2(F.*O_hat); 
    S_hat = reshape(sum(abs(I).^2.*mask, [1,2]), [K,1]); 

    e = S_hat- S;
    es(itr) = mean(abs(e).^2, 'all');

    %grad_O = sum(conj(F).*MyFFT2(MyIFFT2(F.*O_hat).*mask).*reshape(e,[1,1,K]), 3);
    grad_O = sum(conj(F).*MyFFT2(I.*mask.*reshape(e,[1,1,K])),3);


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
        title('|S_{hat} - S|^2');


        drawnow();
    end
    itr = itr+1;
end


