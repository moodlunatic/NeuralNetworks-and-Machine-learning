%% preprocessing the data set

cities = readtable('GlobalLandTemperaturesByCity.csv');
t = tonndata(e01,false,false);
tpred = tonndata(e01,false,false);



% covert average temperature table to array
temp = table2array(rio(:, 2)); 

% split temp into 80% training and 20% test sets
num_train = round(size(temp,1)*0.8);
Xtrain = temp(1:num_train);
Xpred = temp(num_train+1:end);
% training set is further split into a 80% sub-training set and a 20% validation set
num_val = round(num_train*0.2);
Xval = Xtrain(num_train-num_val+1:end);
Xtrain = Xtrain(1:num_train-num_val);

% plot the preprocessed data set
figure
plot(rio.dt(1:num_train-num_val),Xtrain(1:end),'-')
hold on
plot(rio.dt(num_train-num_val:num_train),[Xtrain(end);Xval],'-')
plot(rio.dt(num_train:end), [Xval(end);Xpred],'-')
hold off
xlabel("Date")
ylabel("Temp")
title("Temperature Variation in Rio De Janeiro from 1870 until 2012")
legend(["Training set" "Validation set" "Test set"])

%% time-series prediction by feedforward neural network 
% train on the sub-train set 
% use the validation set to choose the best model

laglist = [25:5:125];
neuronlist = [30, 50, 70];

Errlist = zeros(length(laglist), length(neuronlist));
sumErr = zeros(length(laglist), length(neuronlist));

iteration = 5;

for i = [1:iteration],
    j=1;
    
    for lag = laglist,
        k=1;
        for neurons = neuronlist,
            [Xtr, Ytr] = getTimeSeriesTrainData(Xtrain, lag);

            % convert the data to a useful format
            ptr = con2seq(Xtr);
            ttr = con2seq(Ytr);

            % creation of networks
            net1=feedforwardnet(neurons,'trainlm');

            % training and simulation
            net1.trainParam.epochs = 50;
            net1=train(net1,ptr,ttr, 'useParallel', 'yes', 'showResources', 'yes'); 

            datapredict = [];
            datapredict(1,:) = Xtrain(end-lag+1:end,:)';
            predictresult = Xtrain(end-lag+1:end,:)';

            for i = 1:num_val,
                datapredict(i,:) = predictresult(i:end);
                pval = con2seq(datapredict(i,:)');
                tt = sim(net1, pval, 'useParallel', 'yes', 'showResources', 'yes');
                predictresult = [predictresult, cell2mat(tt)];
            end

            predictpart = predictresult(:,lag+1:end)';

            err = mse(predictpart, Xval);
            fprintf('The MSE of lag %d and neurons %d is %f \n', lag, neurons, err); 

            Errlist(j, k) = err;
            k = k + 1;
        end
        j = j + 1;
    end
    sumErr = sumErr + Errlist;
end

finErr = sumErr/iteration;

