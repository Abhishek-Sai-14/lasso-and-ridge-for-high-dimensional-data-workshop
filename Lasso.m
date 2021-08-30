
%lasso regression

%,survived,age,sibsp,parch,fare,1,2,3,female,male,C,Q,S
data=csvread('C:\Users\abhishek_sai\Downloads\lasso and ridge codes\dataset\train.csv');
percTn=75;  % for training we use 75% of the data
[TnSetF, TnSetL, TtSetF, TtSetL]=SplitTrainTestSet(data,percTn);  % function called from below

%LASSO
%returns the structure FitInfo, which contains information about the fit of the models, using any of the input arguments in the previous syntaxes.

[lasso_model,stats]=lasso(TnSetF,TnSetL,'CV',10); 

Blasso=[lasso_model(:,stats.Index1SE)];  % It is finding the best value for lambda , 1 standard deviation from the best value
lasso_Predict=TtSetF*Blasso; % testingset of feature * blasso
% we consider boolean so if itt greater than 0.5 it is true orelse false
confusionchart(logical(TtSetL),(lasso_Predict>0.5));  % Diagonal and off-diagonal cells correspond to correctly and incorrectly classified observations, respectively
title("Lasso")
acc=sum((lasso_Predict>0.5)==TtSetL)/length(lasso_Predict)*100 % accuracy
mse=mean((TtSetL - lasso_Predict).^2) % mean square error
mae=mean(TtSetL - lasso_Predict) % mean absolutr error



function[TnSetF, TnSetL, TtSetF, TtSetL]=SplitTrainTestSet(Data,PercTn)
    Feature=Data(:,3:14); % leaving first and 2nd column
    Species=Data(:,2); % 2nd column which means survived or not
    TotalNumSamples=length(Species); 
    NumTnSamp=ceil(TotalNumSamples*PercTn/100); % rounds each element of X to the nearest integer greater than or equal to that element
    Indx=randperm(TotalNumSamples); % basically jumble . returns a row vector containing a random permutation of the integers from 1 to n without repeating elements.
    TnSamples=Indx(1:NumTnSamp); 
    TtSamples=Indx(1+NumTnSamp:end); 
    TnSetF=Feature(TnSamples,:);  % tn for training , tt for testing
    TnSetL=Species(TnSamples,:); 
    TtSetF=Feature(TtSamples,:); 
    TtSetL=Species(TtSamples,:);
end% 
