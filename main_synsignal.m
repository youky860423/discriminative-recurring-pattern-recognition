clear
close all;

%%%%%%load synthetic data%%%
type='1d';
width=20;%pattern length 20 for 1d, 2 for 2d;
datasetname=['synsignal_',type,'.mat'];
load(datasetname);
[F,T,N]=size(X);
win_size=2*width;%setting window size
runs=10;
perc=0.8;
no_train=ceil(perc*N);
no_test=N-no_train;
maxiter=10000;
%%%%%training stage%%%%%%%
option.priorType='conv';%1-'conv',0-'times';
option.conv = 1;%1-'conv',0-'fft';
option.addone = 1;
option.lambda = 1e-4;
option.center=1;
option.csiter=500;
option.ceiter=1000;
option.cperiod=100;
option.dc=0;

if option.addone
    no_para=F*win_size+1;
else
    no_para=F*win_size;
end

for i=1:runs
     permidx(i,:)=randperm(N);
     trainX = X(:,:,permidx(i,1:no_train));
     trainY = Y(permidx(i,1:no_train))';
     for j=1:no_train
        trainX2{j} = X2{permidx(i,j)};
     end
    wini=1e-2*randn(no_para,1);
%     tic;
%     [ w{i},rllharr{i},wnorm{i}] = LocalizationGradUpdate( wini,trainX,trainX2,trainY, maxiter,option);
%      runtime_miml(i)=toc 
     tic;
    [ w{i},rllharr{i},wnorm{i}] = LocOptimization( wini,trainX,trainX2,trainY, maxiter,option);
     runtime_miml(i)=toc;
     %%%%%%%%%%%%testing stage%%%%%%%
    for j=1:no_test
        testX(:,:,j) = X(:,:,permidx(i,end-j+1));
        testX2{j} = X2{permidx(i,end-j+1)};
        testY(j,1) = Y(permidx(i,end-j+1),:);
        testy{j} = y{permidx(i,end-j+1)};
    end
    [TPR(:,i),FPR(:,i),AUC(i),bagacc(i)] = testing( w{i},testX,testX2,testY,testy,option,win_size);
     i
     save(['synsignal',type,'_result_',num2str(option.lambda),'_win_',num2str(win_size),'.mat'],...
         'win_size','w','wnorm','permidx','rllharr','TPR','FPR','AUC','bagacc');
end




