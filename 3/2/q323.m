
clc;
clear;
choles=load("choles_all");
X = choles.p';
mapX = mapstd(X);
[ptrans,ps2] = processpca(mapX,0.001);

rconX=processpca('reverse',ptrans,ps2);
mu = mean(X);
result=zeros(2,21);
[eigenvectors, scores] = pca(mapX);
for nComp=1:1:21
Xhat = scores(:,1:nComp) * eigenvectors(:,1:nComp)';
Xhat = bsxfun(@plus, Xhat, mu);
result(1,nComp)=nComp;
result(2,nComp)=sqrt(mean(mean((mapX-Xhat).^2)));
end
plot(1:21,result(2,:));
xlabel("Dimension");
ylabel("erro");
title("Erreo-dimension");













