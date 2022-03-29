clear,clc
load('breast.mat')
T=trainset(:,[20,30,10,17,18,19,5,28,15,29,16,3,8,25,11])
net=feedforwardnet(10);
net.trainFcn="trainlm" ;
net=train(net,T',labels_train')
%%test
TTest=testset(:,[20,30,10,17,18,19,5,28,15,29,16,3,8,25,11])
test_out=sim(net,TTest');
test_out(test_out>=0.5)=1;
test_out(test_out<0.5)=-1;
rate=sum(test_out==labels_test')/length(labels_test');



