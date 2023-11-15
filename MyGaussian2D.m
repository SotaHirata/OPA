function distribution= MyGaussian2D(nx,ny,mu_x,mu_y,sigma_x,sigma_y,sigma_xy)

if (nargin < 6), sigma_y=sigma_x; end;
if (nargin < 7), sigma_xy=0; end;

[x y]=ndgrid(1:nx,1:ny);

distribution=(1/(2*pi*sigma_x*sigma_y*(1-sigma_xy.^2)^0.5))*...
    exp(-1/(2*(1-sigma_xy.^2))*(((x-mu_x).^2)/(sigma_x.^2)+((y-mu_y).^2)/(sigma_y.^2)-...
    2*sigma_xy*((x-mu_x).*(y-mu_y))/(sigma_x*sigma_y)));