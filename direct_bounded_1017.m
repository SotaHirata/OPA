close all;
clear all;clc;

%乱数seedの固定
rng(0);

%パラメタ
M = 8;     %uniformアレイの1辺の長さ
N = M^2;    %アンテナ数
K =  N^2*4;    %照射パターン数
maskD = N/2; %PDの受光範囲の直径

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
S = reshape(sum(abs(MyIFFT2(MyFFT2(exp(1i*(phi+r)).*array).*obj)).^2.*mask, [1,2]), [K,1]); 

%Sを折れ線グラフにプロット
figure;
plot(S,'LineWidth',2);
xlabel('Number of patterns K', 'FontSize', 14); 
ylabel('PD Intensity', 'FontSize', 14); 
Title = sprintf('Uniform array (N=%d, K=%d, MaskDiameter=%d)', N, K, maskD);
title(Title, 'FontSize', 16) ; 
drawnow;
