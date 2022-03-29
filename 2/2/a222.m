[idx1,Centers]=kmeans(labels_train,2)
output1= zeros(400,2) ;
for i = 1 : 400
 output1( i ,idx1(i) ) = 1 ;
end
[idx2,Centers]=kmeans(labels_test,2)
output2= zeros(169 ,2) ;
for j = 1 : 169
 output2( j ,idx2(j) ) = 1 ;
end


[input,PS] =mapminmax(trainset');
testInput = mapminmax (testset')
output1=output1'
output2=output2'

%%
 cvFold = crossvalind('Kfold',length(input),15);       
                    H_net_tmp = newff(input,output1,[4,3],{'logsig','tansig'},'traingd');%PH为输入数据矩阵（必须是标准化后的）
                  
                    H_net_tmp.trainParam.epochs = 10000;%最大迭代次数，我设置的是50000
            
                    H_net_tmp.divideFcn = 'divideind';
              
                    epoch_N = [];%用于存储没戏最后的训练次数
                    for n = 1 : 10
                        testIdx = (cvFold == n);
                        trainIdx = ~testIdx;
                        trInd = find(trainIdx);
                        tstInd = find(testIdx);
                        
                        H_net_tmp.divideParam.trainInd = trInd;
                        H_net_tmp.divideParam.testInd = tstInd;
                        
                        %training Network
                        %reconfigured assignment of random initial weights
                        H_net_tmp = init(H_net_tmp);%每次都对权重进行初始化，很重要，网上的很多代码都没有说明这点
                        [H_net_tmp,tr] = train(H_net_tmp,input,output1);
                        [~,I] = min(tr.tperf);
                        I = I(end);
                        epoch_N(n) = I;%I就是滴k次训练最优的那个次数
                    end
                    epoch_final = ceil(mean(epoch_N));







%%
an=sim(H_net_tmp,testInput)
error=output2-an

[s1 , s2] = size( an ) ;
hitNum = 0 ;
for i = 1 : s2
    if round((an( : ,  i ) ))==(output2(:,i)) ;
        hitNum = hitNum + 1 ; 
    end
end
sprintf('the recognition rate is  %3.3f%%',100 * hitNum / s2 )




