Tnew=(9*T1+8*T2+5*T3+T4+T5)/24;
trainNum=randperm(13600,1000);
valNum=randperm(13600,1000);
testNum=randperm(13600,1000);
X11=X1';
X22=X2';
Tnewn=Tnew';
%%train set 
trX1=X11(:,trainNum);
trX2=X22(:,trainNum);
trXX=[trX1;trX2];
trNew=Tnewn(:,trainNum);
%%val set 
valX1=X11(:,valNum);
valX2=X22(:,valNum);
valXX=[valX1;valX2];
valNew=Tnewn(:,valNum);
%%test set 
testX1=X11(:,testNum);
testX2=X22(:,testNum);
testXX=[testX1;testX2];
testNew=Tnewn(:,testNum);
%surface
[Tr_X1,Tr_X2]=meshgrid(0:0.1:1,0:0.1:1)
Tr_Tnew=griddata(trX1,trX2,trNew,Tr_X1,Tr_X2)
surf(Tr_X1,Tr_X2,Tr_Tnew)
title('surface of training set')
n=25
%% create the net
net=newff(trXX,trNew,n,{'tansig','purelin'},'trainlm','learngdm')
net.trainParam.epochs=10000;
%% train the net
net=train(net,trXX,trNew)
%% validate the net
re=sim(net,testXX)
error2=testNew-re
[T_X1,T_X2]=meshgrid(0:0.1:1,0:0.1:1)
pta=griddata(testX1,testX2,re,T_X1,T_X2)
surf(T_X1,T_X2,pta)
grid on
box on 
title('surface of test set')
RMSE1=(mse(error2*error2'))^0.5
plot(error2)
title('error of test set')
%% val the net
re=sim(net,valXX)
error1=valNew-re
[V_X1,V_X2]=meshgrid(0:0.1:1,0:0.1:1)
pva=griddata(valX1,valX2,re,V_X1,V_X2)
surf(V_X1,V_X2,pva)
grid on
box on 
title('surface of validation set')
RMSE2=(mse(error1*error1'))^0.5







