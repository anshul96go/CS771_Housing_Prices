k=xlsread('train',1,'A1:BT1452');
y=k(1:1162,20);
yt=k(1162:1451,20);
x=[ones(1162,1),k(1:1162,1:19),k(1:1162,21:72)];
xt=[ones(290,1),k(1162:1451,1:19),k(1162:1451,21:72)];

l=[0:0.1:5,5:1:100];
mse=zeros(size(l,2),1);
r=zeros(size(l,2),1);
for i=1:1:size(l,2)
b=(x'*x+l(i)*eye(72))^-1*x'*y;
ssr=sum((x*b-yb*(ones(1162,1))).^2);
sst=sum((y-yb*(ones(1162,1))).^2);
r(i)=ssr/sst;
mse(i)=sqrt(sum((xt*b-yt).^2)/290);
end
yb=mean(y);
a=figure; 
a=plot(l,mse);
xlabel('lambda')
ylabel('Root Mean Square Error')
b=figure; 
b=plot(l,r);
xlabel('lambda')
ylabel('R Squared')
