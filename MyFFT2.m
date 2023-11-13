function y = MyFFT2(x)

y=fftshift(fftshift(fft2(ifftshift(ifftshift(x,1),2)),1),2);