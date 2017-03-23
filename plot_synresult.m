clear;
close all;

%%%%%%%%loading trained convolutional kernal and data%%%%%%%%
load('synsignal_2d.mat');
[F,T,N]=size(X);

option.priortype=1;%1-'conv',0-'times';
option.conv = 1;
option.addone=1;
perc=0.8;
no_train=ceil(perc*N);
no_test=N-no_train;
runs=10;
%%%%%obtaining result files%%%%%%
files=dir('synsignal2d_result*.mat');
for f=1:length(files)
    filename=files(f).name;
    idx1=strfind(filename,'result_');
    idx2=strfind(filename,'_win_');
    lambda=str2double(filename(idx1+7:idx2-1))
    win_size=str2double(filename(idx2+5:end-4))
    width=win_size;

    for i=1:runs
        load(['synsignal2d_result_',num2str(lambda),'_win_',num2str(win_size),'.mat']);
        %%%%%%%%%%%%display learned discriminative pattern%%%%%%
        figure(1)
        if option.addone
            words=reshape(w{i}(1:end-1),width,[]);
        else 
            words=reshape(w{i},width,[]);
        end
        if F>1
            imagesc(words');colormap gray; colorbar;
        else
            plot(words')
        end
        title('conv kernal');
        if option.priortype
            wtxold=WTX(X,w{i},option.addone,option.conv);
        else
            for b=1:B
                wtxold(:,b)=X2{b}'*wold;
            end
        end
        prior=zeros(size(wtxold));
        prior(wtxold<0)=exp(wtxold(wtxold<0))./(1+exp(wtxold(wtxold<0)));
        prior(wtxold>=0)=1./(1+exp(-wtxold(wtxold>=0)));
        prior=prior(ceil(winsize/2):end-floor(winsize/2),:);
        tmpidx=randi(no_train,5,1);
        figure(2)
        for aa=1:5
            prior_2c(1,:,aa)=prior(:,tmpidx(aa));
            prior_2c(2,:,aa)=1-prior(:,tmpidx(aa));
            subplot(5,1,aa);imagesc(prior_2c(:,:,aa));
            title('localization signals');
        end
        figure(3)
        for aa=1:5
            if F>1
                subplot(5,1,aa);imagesc(X(:,:,tmpidx(aa)));
            else
                subplot(5,1,aa);plot(X(:,:,tmpidx(aa)));
            end
            title(['data with label ', num2str(Y(tmpidx(aa)))]);
        end
        figure(4)
        for aa=1:5
            subplot(5,1,aa);imagesc(y{tmpidx(aa)});
            title('ground truth');
        end
        figure(5)
        plot(FPR(:,i),TPR(:,i),'bo-')
        title('ROC')  
        pause()
    end
end
