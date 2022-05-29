%-------------------------------------------------------------------------%
%  Genetic Algorithm (GA) source codes demo version                       %
%                                                                         %
%  Programmer: Jingwei Too                                                %
%                                                                         %
%  E-Mail: jamesjames868@gmail.com                                        %
%-------------------------------------------------------------------------%

% Benchmark data set contains 351 instances and 34 features (binary class)
%------------------------------load ionosphere.mat; % Matlab also provides this dataset (load Ionosphere.mat)
% Call features & labels
feat=X;
label=Y; 
%---Input------------------------------------------------------------------
% feat:  feature vector (instances x features)
% label: labelling
% N:     Number of chromosomes
% T:     Maximum number of generations
% CR:    Crossover rate
% MR:    Mutation rate
% *Note: k-value of KNN & k-fold setting can be modified in jFitnessFunction.m
%---Output-----------------------------------------------------------------
% sFeat: Selected features (instances x features)
% Sf:    Selected feature index
% Nf:    Number of selected features
% curve: Convergence curve
%--------------------------------------------------------------------------
%% (Method 1) GA  
close all; N=10; T=100; CR=0.8; MR=0.01; 
[sFeat,Sf,Nf,curve]=jGA(feat,label,N,T,CR,MR);
% Plot convergence curve
figure(); plot(1:T,curve); xlabel('Number of Iterations');
ylabel('Fitness Value'); title('GA'); grid on;
%% (Method 2) GA version 2 
% Assume the chromosomes with CR probabilities are used in selection process 
close all; N=10; T=100; CR=0.8; MR=0.01; 
[sFeat,Sf,Nf,curve]=jGA2(feat,label,N,T,CR,MR);
% Plot convergence curve
figure(); plot(1:T,curve); xlabel('Number of Iterations');
ylabel('Fitness Value'); title('GA2'); grid on;
