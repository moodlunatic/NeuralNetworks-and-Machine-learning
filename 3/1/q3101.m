close all;
clear all;
clc;
banana=load('banana.mat') ;
T=(banana.X)';
net = newsom(T,[16 16],'hextop','linkdist'); 

% plot the data distribution with the prototypes of the untrained network
figure;plot(T(1,:),T(2,:));
hold on
plotsom(net.iw{1},net.layers{1}.distances)
hold off

% finally we train the network and see how their position changes
net.trainParam.epochs = 100;
net = train(net,T);
figure;plot(T(1,:),T(2,:));
hold on
plotsom(net.iw{1},net.layers{1}.distances)
hold off


