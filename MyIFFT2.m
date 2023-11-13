function y = MyIFFT2(x)

y=fftshift(fftshift(ifft2(ifftshift(ifftshift(x,1),2)),1),2);