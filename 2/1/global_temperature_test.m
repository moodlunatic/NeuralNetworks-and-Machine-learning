%% preprocessing the data set

cities = readtable('GlobalLandTemperaturesByCity.csv');
% select the city to analyze, e.g. 'Rio De Janeiro'
idx_cities = find(string(cities.City) == 'Rio De Janeiro');
rio = cities(idx_cities, :);
% Preserve only the dates and average temperatures
rio = rio(:, [1,2]);
% convert date to type datetime
rio.dt = datetime(rio.dt);
% consider the temperature just from 1870 until the end of 2012
idx_time = find(year(rio.dt) >= 1870 & year(rio.dt) < 2013);
rio = rio(idx_time, :);


% covert average temperature table to array
temp = table2array(rio(:, 2)); 

% split temp into 80% training and 20% test sets
num_train = round(size(temp,1)*0.8);
Xtrain = temp(1:num_train);
Xpred = temp(num_train+1:end);


%% time-series prediction by feedforward neural network 
% use the chosen lag and neuron to build the nueral network again on the whole training set
% assume that the lag chosen is 50 and the neurons chosen is 50

lag = 50;
neurons = 50;


% build and train the network once again
[Xtr, Ytr] = getTimeSeriesTrainData(Xtrain, lag);

% convert the data to a useful format
ptr = con2seq(Xtr);
ttr = con2seq(Ytr);

% creation of networks
net1=feedforwardnet(neurons,'trainlm');

% training and simulation
net1.trainParam.epochs = 50;
net1=train(net1,ptr,ttr, 'useParallel', 'yes', 'showResources', 'yes'); 
            

%% Test on the test set

datapredict_test = [];
datapredict_test(1,:) = Xtrain(end-lag+1:end,:)';
predictresult_test = Xtrain(end-lag+1:end,:)';
num_test = size(Xpred,1)

for i = 1:num_test,
    datapredict_test(i,:) = predictresult_test(i:end);
    ptest = con2seq(datapredict_test(i,:)');
    tt_test = sim(net1, ptest, 'useParallel', 'yes', 'showResources', 'yes');
    predictresult_test = [predictresult_test, cell2mat(tt_test)];
end

predictpart_test = predictresult_test(:,lag+1:end)';

err = mse(predictpart_test, Xpred);
fprintf('The MSE of lag %d and neurons %d is %f \n', lag, neurons, err); 


% plot the prediction
figure
hold on
plot(rio.dt(num_train+1:end), Xpred,'-')
plot(rio.dt(num_train+1:end), predictpart_test,'-')
hold off

xlabel("Date")
ylabel("Temp")
title("Prediction on Temperature Variation in Rio De Janeiro")
legend(["Test set" "Prediction"])        