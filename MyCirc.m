function y= MyCirc(n,m,center)

if (nargin < 3), center=floor(n/2)+1; end;

if(size(n)==1), n(2)=n(1); end;
if(size(m)==1), m(2)=m(1); end;
if(size(center)==1), center(2)=center(1); end;

y=zeros(n);

[i j]=ndgrid(1:n(1),1:n(2));

y((i-center(1)).^2/(m(1)/2).^2+(j-center(2)).^2/(m(2)/2).^2<=1)=1;
