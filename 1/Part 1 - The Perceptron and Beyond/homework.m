x=linspace(0,1,21);
y=-sin(0.8*pi*x);
plot(x,y);
hold on;
%linear model
P=x;
T=y;
lr=maxlinlr(P,'bias');
net=linearlayer(0,lr);
net=train(net,P,T);
newx=linspace(0,1,10);
newy=sim(net,newx);
plot(newx,newy);

net = fitnet(2);
net = configure(net,x,y);
net.inputs{1}.processFcns = {};
net.outputs{2}.processFcns = {};
[net, tr] = train(net,x,y);

