Tnew=(8*T1+6*T2+4*T3+2*T4+2*T5)/22;
id=randperm(13600)
X1=X1(id(1:1000),:)
X2=X2(id(1:1000),:)
Tnew=Tnew(id(1:1000),:)
X1=X1'
X2=X2'
Tnew=Tnew'
%% training set
tr_X1=X1(:,(1:500))
tr_X2=X2(:,(1:500))
tr_X=[tr_X1;tr_X2]
tr_Tnew=Tnew(:,(1:500))
%% val set 
v_X1=X1(:,(501:750))
v_X2=X2(:,(501:750))
v_X=[v_X1;v_X2]
v_Tnew=Tnew(:,(501:750))
%% test set
t_X1=X1(:,(751:1000))
t_X2=X2(:,(751:1000))
t_X=[t_X1;t_X2]
t_Tnew=Tnew(:,(751:1000))

%% surface plot
[Tr_X1,Tr_X2]=meshgrid(0:0.1:1,0:0.1:1)
Tr_Tnew=griddata(tr_X1,tr_X2,tr_Tnew,Tr_X1,Tr_X2)
surf(Tr_X1,Tr_X2,Tr_Tnew)
grid on
box on 
title('surface of training set')
%% create the net
net=newff(tr_X,tr_Tnew,30,{'logsig','purelin'},'trainlm','learngdm')
net.trainParam.epochs=10000;
%% train the net
net=train(net,tr_X,tr_Tnew)
%% validate the net
an=sim(net,v_X)
error1=v_Tnew-an
[V_X1,V_X2]=meshgrid(0:0.1:1,0:0.1:1)
An=griddata(v_X1,v_X2,an,V_X1,V_X2)
surf(V_X1,V_X2,An)
grid on
box on 
title('surface of validation set')
RMSE = sqrt(MSE)



