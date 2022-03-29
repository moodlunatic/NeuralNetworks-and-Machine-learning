
clc;
clear;
X = randn(50,500)';
mu = mean(X);
result=zeros(2,50);
[eigenvectors, scores] = pca(X);
for nComp=1:1:50
Xhat = scores(:,1:nComp) * eigenvectors(:,1:nComp)';
Xhat = bsxfun(@plus, Xhat, mu);
result(1,nComp)=nComp;
result(2,nComp)=sqrt(mean(mean((X-Xhat).^2)));
end
plot(1:50,result(2,:));
xlabel("Dimension");
ylabel("erro");
title("Erreo-dimension");










