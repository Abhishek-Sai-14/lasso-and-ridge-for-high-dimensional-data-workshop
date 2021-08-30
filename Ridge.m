%,survived,age,sibsp,parch,fare,1,2,3,female,male,C,Q,S
%ridge regression
%,survived,age,sibsp,parch,fare,1,2,3,female,male,C,Q,S

data=csvread('C:\Users\sai teja\Downloads\lasso and ridge codes\dataset\train.csv');
 M=891; % number of rows
 N=12;  % number of columns
percTn=75;
[TnSetF, TnSetL, TtSetF, TtSetL]=SplitTrainTestSet(data,percTn); % calling the function from below

%RIDGE
l=1e-3;
Bridge=ridge(TnSetL,TnSetF,l); % , the function computes ridge after centering and scaling the predictors to have mean 0 and standard deviation 
ridge_Predict=TtSetF*Bridge>0.5; % we take only which are true
mse=mean((TtSetL - ridge_Predict).^2) % mean square error
mae=mean(TtSetL - ridge_Predict) % mean absolute error
confusionchart(logical(TtSetL),ridge_Predict); % Diagonal and off-diagonal cells correspond to correctly and incorrectly classified observations, respectively
title("Ridge")
acc=sum(ridge_Predict==TtSetL)/length(ridge_Predict) % accuracy

function[TnSetF, TnSetL, TtSetF, TtSetL]=SplitTrainTestSet(Data,PercTn)
    Feature=Data(:,3:14); % leaving first and 2nd column
    Species=Data(:,2); % 2nd column which means survived or not
    TotalNumSamples=length(Species); 
    NumTnSamp=ceil(TotalNumSamples*PercTn/100); % rounds each element of X to the nearest integer greater than or equal to that element
    Indx=randperm(TotalNumSamples); % basically jumble . returns a row vector containing a random permutation of the integers from 1 to n without repeating elements.
    TnSamples=Indx(1:NumTnSamp); 
    TtSamples=Indx(1+NumTnSamp:end); 
    TnSetF=Feature(TnSamples,:); 
    TnSetL=Species(TnSamples,:); 
    TtSetF=Feature(TtSamples,:); 
    TtSetL=Species(TtSamples,:);
end% 