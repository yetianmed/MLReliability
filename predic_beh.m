clear
close all
load example_data.mat
% xTrain: FC matrices of train set: nSubjects x nEdges
% xTest: FC matrics of test set: nSubjects x nEdges
% yTrain: fIQ of train set: nSubjects x 1
% yTest: fIQ of test set: nSubjects x 1
% covariate_train=[age,sex,FD]; % train
% covariate_test=[age,sex,FD]; % test

% ==== Set some parameters for prediction ==== %
penalty='ridge';%'lasso','ridge''kridge';

%Range for lambda
L='default';

%Max number lambda values evaluated
J=100;

%Number of cross-validations
ncvals=20;

%Search strategy {'gridsearch','bayes'}
opt_method='gridsearch';

%Show a plot of the error as a function of lambda
ShowPlot=0;

if strcmp(penalty,'kridge')
    nSubjects=size(xTrain,1);
    coef=zeros(nSubjects,2); %beta coeffients
else
    coef=zeros(size(xTrain,2),2); % number of edges x 2 groups Coefficients of edges
end

%loop over the two independent samples
r=zeros(2,1); % Prediction accuracy
coef0=zeros(2,1);
lambda=zeros(2,1);
yhat=zeros(size(yTrain,1),2); % Save yhat
for j=1:2
    
    if j==1
        % Train is used to train the model and test is used to test
        x=xTrain; y=yTrain;
        xtest=xTest; ytest=yTest;
        
        % Confounds regression
        [~,~,STATS] = glmfit(covariate_train,y);
        y = STATS.resid; % Residuals
        
        %Fit the confounds regression coefficients to test sample
        yfit = glmval(STATS.beta,covariate_test,'identity'); % Fit the model to test
        ytest = ytest-yfit; % Residuals of y in test
        
    elseif j==2
        % Test is used to train the model and train is used to test
        x=xTest; y=yTest;
        xtest=xTrain; ytest=yTrain;
        
        % Confounds regression
        [~,~,STATS] = glmfit(covariate_test,y);
        y = STATS.resid; % Residuals
        
        % Fit the confounds regression coefficients to test sample
        yfit = glmval(STATS.beta,covariate_train,'identity'); % Fit the model to test
        ytest = ytest-yfit; % Residuals of y in test
    end
    
    % Fit model
    if strcmp(penalty,'lasso')
        solver={'sparsa'};
    elseif strcmp(penalty,'ridge')
        solver={'asgd','lbfgs'};
    elseif strcmp(penalty,'kridge')
        solver={'asgd','lbfgs'};
        xtest=corr(xtest',x'); %subject x subject %Reference back to kernel
        x=corr(x');%subject x subject
        penalty='ridge';
    end
    
    params=hyperparameters('fitrlinear',x,y);
    
    if ~strcmp(L,'default')
        params(1).Range=L;
    end
    
    Repeat=1;    %Is a repeat necessary? 1=yes, 0=no
    RepeatCnt=0; %No more than two repeat attempts
    
    while Repeat && RepeatCnt<2
        
        params(2).Optimize=false; params(3).Optimize=false;
        
        % cognitive measures
        [mdl,~,hyper]=fitrlinear(x',y,... % transpose x->x':significant reduction in optimization-execution time
            'Learner','leastsquares',...
            'Regularization',penalty,...
            'OptimizeHyperparameters',params,...
            'Verbose',0,...
            'Solver',solver,...  %default is 'sgd'
            'ObservationsIn','columns',... % transpose x->x':significant reduction in optimization-execution time
            'PostFitBias',true,...         %default is false
            'PassLimit',10,...             %default is 1 (increasing will increase run time)
            'HyperparameterOptimizationOptions',struct('Kfold',ncvals,...
            'Optimizer',opt_method,...
            'NumGridDivisions',J,...
            'ShowPlots',false,...
            'Repartition',false,...
            'MaxObjectiveEvaluations',J));
             
        coef(:,j)=mdl.Beta; coef0(j)=mdl.Bias; lambda(j)=mdl.Lambda;
        
        % Evaluate model in testing sample
        yhat(:,j) = xtest*coef(:,j) + coef0(j);
        
        %plot the error as a function of lambda
        if ShowPlot && exist('hyper','var') && strcmp(opt_method,'gridsearch')
            w=table2array(hyper);
            [~,ind_srt]=sort(w(:,1));
            figure; semilogx(w(ind_srt,1),w(ind_srt,2)); xlabel('Lambda'); ylabel('Error');
        end
        
        if exist('hyper','var') && strcmp(opt_method,'gridsearch')
            w=table2array(hyper);
            [~,ind_srt]=sort(w(:,1));
            if sum(coef(:,j))==0 %null solution
                mse_vals=w(ind_srt,2);
                lam_vals=w(ind_srt,1);
                cnt=0;
                while abs(mse_vals(end-cnt)-mse_vals(end-cnt-1))<0.00000001
                    cnt=cnt+1;
                end
                new_L_end=lam_vals(end-cnt-1);
                fprintf('Repeating with a truncated Lambda of %0.2f\n',new_L_end);
                params(1).Range(2)=new_L_end;
                RepeatCnt=RepeatCnt+1;
            else
                Repeat=0;
            end
        else
            Repeat=0;
        end
    end
    
    % Prediction accuracy
    r(j)=corr(ytest,yhat(:,j));
    fprintf('r=%0.3f\n',r(j));
end
fprintf('\nMACHINE LEARNING,%s\n',penalty);
fprintf('correlation Test -> Train: %0.3f\n',r(1));
fprintf('correlation Train -> Test: %0.3f\n',r(2));

%ICC between beta coefficients
icc_coef=ICC(zscore(coef),'1-1');
fprintf('correlation between betas: %0.3f\n',icc_coef);





