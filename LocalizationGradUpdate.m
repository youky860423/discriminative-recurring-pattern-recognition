function [ wnew,rllharr,wnorm] = LocalizationGradUpdate( w,X,X2,Y,maxiter,option)
%LOCALIZATIONGRADUPDATE Summary of this function goes here
%   input: X in (F, T, B), where F is the number of frequence bins if X is
%   1-d signal, set F to 1, T is the number of time samples, B is total
%   number of training examples. w in (no_parameter,1) is the
%   distriminative classifier. maxiter is the maximum number of
%   iterations, option is setting a model methods.
%   output: wnew is the learned classifier, rllharr is the output negative
%   likelihood versus number of iteration, garr is the upperbound versus
%   number of iteration.
rllharr=zeros(maxiter,1);
wnorm=zeros(maxiter,1);
step=1;
if option.addone
    width=size(w(1:end-1),1)/size(X,1);
else
    width=size(w,1)/size(X,1);
end
F=size(X,1);
B=size(X,3);
for iter=1:maxiter
    wnorm(iter)=norm(w(1:end-1));
    wold = w;
    if strcmp(option.priorType,'conv')
        wtxold=WTX(X,wold,option.addone,option.conv);
    else
        for b=1:B
            wtxold(:,b)=X2{b}'*wold;
        end
    end
    wtxold_max=max(wtxold,[],1);
    wtxold_minus_max=wtxold-ones(size(wtxold,1),1)*wtxold_max;
    post=exp(wtxold_minus_max)./(ones(size(wtxold,1),1)*sum(exp(wtxold_minus_max),1));
    tmp=zeros(size(wtxold));
    tmp(wtxold<0)=log(1+exp(wtxold(wtxold<0)));
    tmp(wtxold>=0)=log(1+exp(-wtxold(wtxold>=0)))+wtxold(wtxold>=0);
    rll_part2=log(sum(exp(wtxold_minus_max),1))+wtxold_max;
    rllharr(iter)=sum(tmp(:))-sum(rll_part2.*Y);
    step=10*step;
    [w, step, ~]=MaximizationStep(X,X2,Y,wold,wtxold,post,step,option);
    %w(1:end-1)=w(1:end-1)-mean(w(1:end-1)); 
    %%%%%%%%shift%%%%%%%
     if option.center
         if iter>=option.csiter && iter<=option.ceiter && rem(iter,option.cperiod)==0
            if option.addone
                words=reshape(w(1:end-1),width,[]);
            else 
                words=reshape(w,width,[]);
            end
            figure(1)
            if F>1
                imagesc(words');colormap gray; colorbar;
            else
                plot(words');
            end
            title(['words before center in iter ',num2str(iter)]);
            [~,idx]=max(sum(words.^2,2));
            wordsnew=circshift(words,ceil(width/2)-idx);
            figure(8)
            if F>1
                imagesc(wordsnew');colormap gray; colorbar;
            else
                plot(wordsnew');
            end
            if option.addone
                w(1:end-1)=wordsnew(:);
            else
                w=wordsnew(:);
            end
         end
     end
    if (rem(iter, 1000)==0)
        iter
% %  %%%%%%%%%%%%display dictionary words at each iteration%%%%%%
% %         figure(1)
% %         if option.addone
% %             words=reshape(w(1:end-1),width,[]);
% %         else 
% %             words=reshape(w,width,[]);
% %         end
% %         if F>1
% %             imagesc(words');colormap gray; colorbar;
% %         else
% %             plot(words')
% %         end
% %         title(['words in iter ',num2str(iter)]);
% %         prior=zeros(size(wtxold));
% %         prior(wtxold<0)=exp(wtxold(wtxold<0))./(1+exp(wtxold(wtxold<0)));
% %         prior(wtxold>=0)=1./(1+exp(-wtxold(wtxold>=0)));
% %         
% %         figure(2)
% %         for aa=1:5
% %             prior_2c(1,:,aa)=prior(:,aa);
% %             prior_2c(2,:,aa)=1-prior(:,aa);
% %             subplot(5,1,aa);imagesc(prior_2c(:,:,aa));
% %             title('prior prob');
% %         end
% %         figure(3)
% %         for aa=1:5
% %             posterior_2c(1,:,aa)=post(:,aa)*Y(aa);
% %             posterior_2c(2,:,aa)=1-post(:,aa)*Y(aa);
% %             subplot(5,1,aa);imagesc(posterior_2c(:,:,aa));
% %             title('posterior prob');
% %         end
% %         figure(4)
% %         for aa=1:5
% %             subplot(5,1,aa);imagesc(prior_2c(:,:,aa)-posterior_2c(:,:,aa),[-1 1]);
% %             title('difference');
% %         end
% %         figure(5)
% %         for aa=1:5
% %             subplot(5,1,aa);
% %             if F>1
% %                 imagesc(X(:,:,aa));colormap gray;
% %             else
% %                 plot(X(1,:,aa));
% %             end
% %             title('data');
% %         end
% %         figure(6)
% %         semilogy(rllharr);
% %         title('negative log likelihood');
% %         figure(7)
% %         semilogy(wnorm);
% %         title('norm of w');
% %         pause(0.1);
    end
end
wnew=w;

end

function [w_new, step, enough]=MaximizationStep(X,X2,Y,w,wtx,post,step_init,option)

grad=gradient(wtx,post,w,X,X2,Y,option);
[w_new,step, enough]=BackTracking(post,w,X,X2,Y,grad,0.5,0.7,step_init,option);
% % options2 = struct('GradObj','on','Display','off','LargeScale','off','HessUpdate','lbfgs','GoalsExactAchieve',0,'MaxIter',1);
% % [w_new,step,enough,output,grad] = fminlbfgs(@(w)gfunc(post,w,X,X2,Y,option),w,options2);

end

function [llh,grad,hess]=gfunc(post_weight,w,X,X2,Y,option)
T=size(X,2);
B=size(X,3);
wtx=zeros(T,B);
if strcmp(option.priorType,'conv')
    wtx=WTX(X,w,option.addone,option.conv);
else
    for b=1:B
        wtx(:,b)=X2{b}'*w;
    end
end
tmp=zeros(size(wtx));
tmp(wtx<0)=log(1+exp(wtx(wtx<0)));
tmp(wtx>=0)=log(1+exp(-wtx(wtx>=0)))+wtx(wtx>=0);
llh=sum(sum(tmp))-sum(sum(post_weight.*wtx,1).*Y);
% % %%%%%adding the constant part%%%%%%%%%%
% % llh=llh+sum((sum(post_weight.*wtxold,1)-log(sum(exp(wtxold_minus_max),1))+wtxold_max).*Y);
% llh=sum(sum(tmp));sum(sum(post_weight.*wtx,1).*Y);
if option.addone
    tmpw=w(1:end-1);
else
    tmpw=w;
      
end
llh=llh+0.5*option.lambda*sum(tmpw.^2);

if nargout > 1
    grad=gradient(wtx,post_weight,w,X,X2,Y,option);
%     grad=grad(:);
end

if nargout > 2
    hess=Hessian(wtx,X2);
end

end

function grad=gradient(wtx,post_weight,w,X,X2,Y,option)
prior=zeros(size(wtx));
prior(wtx<0)=exp(wtx(wtx<0))./(1+exp(wtx(wtx<0)));
prior(wtx>=0)=1./(1+exp(-wtx(wtx>=0)));
Activations=prior-post_weight.*(ones(size(wtx,1),1)*Y);
% Activations=prior;%post_weight.*(ones(size(wtx,1),1)*Y);
F=size(X,1);
B=size(X,3);
if (option.addone)
    width=(size(w,1)-1)/F;
else
    width=size(w,1)/F;
end
tempgrad=zeros(size(w,1),B);
if strcmp(option.priorType,'conv')
    tempgrad = PTW(X,Activations,width,option.addone,option.conv);%conv 1, fft 0
else
    for b=1:B
        tempgrad(:,b) = X2{b}*Activations(:,b);
    end
end
grad = sum(tempgrad,2);
if option.addone
    grad(1:end-1,:) = option.lambda*w(1:end-1,:) + grad(1:end-1,:);
    %%%%%%%adding dc constraint%%%%%%%%%
    if option.dc
        grad(1:end-1,:) = grad(1:end-1,:) - mean(grad(1:end-1,:));
    end
else
    grad = option.lambda*w + grad;
    %%%%%%%adding dc constraint%%%%%%%%%
    if option.dc
        grad = grad - mean(grad);
    end
end

end

function hessian=Hessian(wtx,X2)
prior=zeros(size(wtx));
prior(wtx<0)=exp(wtx(wtx<0))./(1+exp(wtx(wtx<0)));
prior(wtx>=0)=1./(1+exp(-wtx(wtx>=0)));
temphess=zeros(size(w,1),size(w,1),B);
for b=1:B
    temphess(:,:,b) = (X2{b}.*(ones(size(X2{b},1),1)*prior(:,b)'))*...
                (((1-prior(:,b))*ones(1,size(X2{b},1))).*X2{b});
end
hessian = sum(temphess,3);

end
%%%%%backtracking line search%%%%%
function [w_new, step, enough]=BackTracking(post,w,X,X2,Y,grad,alpha,beta,step_init,option)
stop=0;
enough=0;
f=gfunc(post,w,X,X2,Y,option);
f1=grad(:);
step=step_init;
% % %%%%%%%%%%%%debugging for gradient and objective%%%%%%
% % cnt=0;
% % rng=linspace(0,step_init,1000);
% % for step2=rng
% %     cnt=cnt+1;
% %     w_step2=w-step2*grad;
% %     f_w_step2(cnt)=gfunc(post,w_step2,X,X2,Y,option);
% %     f_w_step3(cnt)=f-step2*(f1'*f1);
% %     f_w_step4(cnt)=f-alpha*step2*(f1'*f1);
% % end
% % %w2=w;
% % eps88=1e-8;
% % % tmp7=[randn(size(w(1:end-1)));0];
% % % tmp7=[zeros(size(w(1:end-1)));randn(1)];
% % tmp7=randn(size(w));
% % w2=w+eps88*tmp7;
% % f2=gfunc(post,w2,X,X2,Y,option);
% % ((f2-f)/eps88 - sum(grad.*tmp7))/(sum(grad.*tmp7))
% % figure(6)
% % plot(rng,f_w_step2,'b')
% % hold on 
% % plot(rng,f_w_step3,'r')
% % plot(rng,f_w_step4,'g')
% % hold off
% % title('every iteration Q');
% % pause()
%%%%%%%%%%%%debugging for gradient and objective%%%%%%
while(stop==0)
    w_step=w-step*grad;
    f_w_step=gfunc(post,w_step,X,X2,Y,option);
    if(f_w_step<f-alpha*step*(f1'*f1))     
            stop=1;
    else
    step=step*beta;
    if ((step<1e-12)&&(f_w_step<=f))
        stop=1;
        enough=1;
    end
    end
end
w_new=w_step;
end
