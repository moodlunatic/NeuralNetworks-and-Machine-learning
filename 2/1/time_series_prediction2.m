load('lasertrain.dat');

%% split temp into 80% training and 20% test sets
num_train = round(size(lasertrain,1)*0.8);
Xtrain = lasertrain(1:num_train);
Xval = lasertrain(num_train+1:end);

laglist =35;
neuronlist =30;

Errlist = zeros(length(laglist),length(neuronlist));
sumErr = zeros(length(laglist),length(neuronlist));

iteration = 5;

for it = [1:iteration],
    j=1;
    
    for lag = laglist,
        k=1;
        for neurons = neuronlist;
            [Xtr,Ytr] = getTimeSeriesTrainData(Xtrain, lag);
                        
            % convert the data to a useful format
            ptr = con2seq(Xtr);
            ttr = con2seq(Ytr);
            
            %creation of networks
            net1=feedforwardnet(neurons,'trainlm');
            
            %training and simulation
            net1.trainParam.epochs = 50;
            net1=train(net1,ptr,ttr); 
           
            datapredict = [];
            datapredict(1,:) = Xtrain(end-lag+1:end,:)';
            predictresult = Xtrain(end-lag+1:end,:)';
            
            for i = 1:200,
                datapredict(i,:) = predictresult(i:end);
                ptest = con2seq(datapredict(i,:)');
                tt = sim(net1, ptest);
                predictresult = [predictresult, cell2mat(tt)];
            end
                
            predictpart = predictresult(:,lag+1:end)';
            
            err = mse(predictpart,Xval);
            fprintf('The MSE of lag %d and neurons %d is %f \n', lag, neurons, err); 
            
% figure
%             plot(predictpart)
%             hold on;
%             plot(laserpred);
%             legend('prediction','test data');
%             title(['Time series prediction results on test data of lag = ',...
%                num2str(lag), ' and neurons = ', num2str(neurons)]);
  
            Errlist(j, k) = err;
            k = k + 1;
        end
        j = j + 1;
    end
    sumErr = sumErr + Errlist;
end

finErr = sumErr/iteration;
%% test set 
load('laserpred.dat');
Xtest=laserpred
datapredict1 = [];
            datapredict(1,:) = Xtrain(end-lag+1:end,:)';
            predictresult1 = Xtrain(end-lag+1:end,:)';
            
            for i = 1:100,
                datapredict(i,:) = predictresult1(i:end);
                ptest = con2seq(datapredict(i,:)');
                tt = sim(net1, ptest);
                predictresult1 = [predictresult1, cell2mat(tt)];
            end
                
            predictpart1 = predictresult1(:,lag+1:end)';
            
            err = mse(predictpart1,Xtest);
            fprintf('The MSE of lag %d and neurons %d is %f \n', lag, neurons, err); 
%% plot
% plot the prediction
figure
hold on
plot([1:100],Xtest,'-')
plot([1:100],predictpart1,'-')
hold off

xlabel("Discrete points")
title("Prediction of Laser")
legend(["Test set" "Prediction"])      
