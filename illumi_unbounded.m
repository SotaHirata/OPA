close all;
clear all;clc;

%乱数seedの固定
rng(0);

%パラメタ
M = 6;     %uniformアレイの1辺の長さ
N = M^2;    %アンテナ数
K = N^2*4;    %照射パターン数

%アンテナ位置を表す行列（N×N）
array = MyRect(N, M); %for uniformアレイ
%load('Costasarray_N127.mat') ;
%array = matrix;%for Costasアレイ

%ランダムな固定された初期位相（0〜2pi）
r = array.*ones(N,N)*2*pi; %for uniformアレイ

%ランダムな位相シフト（0〜2pi）Kパターン
phi = array.*rand(N,N,K)*2*pi;

%照射パターン（N×N×K）
F = MyFFT2(exp(1i*(phi+r)).*array);

% FをF_uni_N(N).matとして保存
varname = sprintf('F_r_phik_uni_N%d_K%d.mat',N, K);
save(varname, 'F', 'r', 'phi');