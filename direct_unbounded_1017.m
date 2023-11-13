close all;
clear all;clc;

%乱数seedの固定
rng(0);

%パラメタ
M = 6;     %uniformアレイの1辺の長さ
N = M^2;    %アンテナ数
K = N^2;    %照射パターン数

%強度分布画像を生成（N×N）
img = imread('peppers.png');
img_resized = imresize(img, [N, N]);
img_gray = double(rgb2gray(img_resized)) ;
obj = img_gray / max(img_gray(:));

%アンテナ位置を表す行列（N×N）
array = MyRect(N, M); %for uniformアレイ
%load('Costasarray_N127.mat') ;
%array = matrix;%for Costasアレイ

%ランダムな固定された初期位相（0〜2pi）
%r = array.*rand(N,N)*2*pi; %for uniformアレイ

%ランダムな位相シフト（0〜2pi）Kパターン
%phi = array.*rand(N,N,K)*2*pi;

%照射パターン（N×N×K配列）
load('F_uni_N36.mat'); %Fに照射パターンを格納
%PDの観測強度（K×1配列）
S = reshape(sum(abs(F).^2.*obj, [1,2]), [K,1]); 

%Sを折れ線グラフにプロット
figure;
plot(S,'LineWidth',2);
xlabel('Number of patterns K', 'FontSize', 14); 
ylabel('PD Intensity', 'FontSize', 14); 
Title = sprintf('Uniform array (N=%d, K=%d, unmasked)', N, K);
title(Title, 'FontSize', 16); 
drawnow;