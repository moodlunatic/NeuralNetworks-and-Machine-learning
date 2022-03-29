clear,clc
load('breast.mat')
net=feedforwardnet(10);
net.trainFcn="trainlm" ;
net=train(net,trainset',labels_train')
%%test
test_out=sim(net,testset');
test_out(test_out>=0.5)=1;
test_out(test_out<0.5)=-1;
rate=sum(test_out==labels_test')/length(labels_test');

[coeff,score,latent,tsquared,explained] = pca(trainset);
explained
scatter(score(:,1),score(:,2))
axis equal
xlabel('1st Principal Component')
ylabel('2nd Principal Component')


rng default % for reproducibility
Y = tsne(trainset,'Algorithm','barneshut','NumPCAComponents',2);
gscatter(Y(:,1),Y(:,2),labels_train)
