% テキストファイルの読み込み
filename = 'Costas_essense_N=41.txt'; % ファイル名を適切に変更
data = importdata(filename);

% Nの計算
N = numel(data);

% N×Nのゼロ行列の作成
matrix = zeros(N, N);

% 各行の対応する位置に1を配置
for i = 1:N
    col = data(i);
    matrix(i, col) = 1;
end

% 結果を.matファイルに保存
outputFilename = 'Costasarray_N41.mat'; % 保存するファイル名を指定
save(outputFilename, 'matrix');

disp(['結果を ' outputFilename ' に保存しました。']);

% matrixを輝度画像に変換
image = uint8(matrix * 255); % 0を黒(0)、1を白(255)に変換

% 輝度画像の表示
imshow(image);
title('Costas Array Image'); % 画像にタイトルを追加

% 輝度画像をファイルに保存
imageFilename = 'CostasArray_N41.png'; % 保存する画像ファイル名を指定
imwrite(image, imageFilename);

disp(['輝度画像を ' imageFilename ' に保存しました。']);