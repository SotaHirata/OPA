function y= MyRect(n,m,center)
%%n×n画素中に, m×mの矩形マスク（中心center）を作る関数（centerはデフォで中心）
if (nargin < 3), center=floor(n/2)+1; end;

if(size(n)==1), n(2)=n(1); end;
if(size(m)==1), m(2)=m(1); end;
if(size(center)==1), center(2)=center(1); end;

y=zeros(n);

i=center-(floor(m/2));

y(i(1):i(1)+m(1)-1,i(2):i(2)+m(2)-1)=1;