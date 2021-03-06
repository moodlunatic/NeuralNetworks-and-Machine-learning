load('lasertrain.dat');
load('laserpred.dat');

laglist = [20:5:125];
neuronlist = [20, 30, 50];

Errlist = zeros(length(laglist),length(neuronlist));
sumErr = zeros(length(laglist),length(neuronlist));

iteration = 1;

for it = [1:iteration],
    j=1;
    
    for lag = laglist,
        k=1;
        for neurons = neuronlist;
            [Xtr,Ytr] = getTimeSeriesTrainData(lasertrain, lag);
                        
            % convert the data to a useful format
            ptr = con2seq(Xtr);
            ttr = con2seq(Ytr);
            
            %creation of networks
            net1=feedforwardnet(neurons,'trainlm');
            
            %training and simulation
            net1.trainParam.epochs = 50;
            net1=train(net1,ptr,ttr); 
           
            datapredict = [];
            datapredict(1,:) = lasertrain(end-lag+1:end,:)';
            predictresult = lasertrain(end-lag+1:end,:)';
            
            for i = 1:100,
                datapredict(i,:) = predictresult(i:end);
                ptest = con2seq(datapredict(i,:)');
                tt = sim(net1, ptest);
                predictresult = [predictresult, cell2mat(tt)];
            end
                
            predictpart = predictresult(:,lag+1:end)';
            
            err = mse(predictpart,laserpred);
            fprintf('The MSE of lag %d and neurons %d is %f \n', lag, neurons, err); 
            
            %figure
            %plot(predictpart)
            %hold on;
            %plot(laserpred);
            %legend('prediction','test data');
            %title(['Time series prediction results on test data of lag = ',...
            %    num2str(lag), ' and neurons = ', num2str(neurons)]);
  
            Errlist(j, k) = err;
            k = k + 1;
        end
        j = j + 1;
    end
    sumErr = sumErr + Errlist;
end

finErr = sumErr/iteration;

            
