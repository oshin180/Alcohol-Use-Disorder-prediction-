%the selected features are in variable 'Sf' of the genetic algorithm code
%the numbers of Sf are:
%2,6,7,21,22,24,25,28,29,30,31,38,39,44,45,46,48,49,51,52,57,59,61

selected_channels=[2,6,7,21,22,24,25,28,29,30,31,38,39,44,45,46,48,49,51,52,57,59,61]; %the Sf is result of GA.
k=0;
p=0;
Ns=256;
numexcelfiles=30;  %file number of the last excel sheet is Data468.csv
currentfolder=cd;
final_wave=[];
wave=[];
 
for i=6:numexcelfiles
    baseFileName='Data';
    extension=num2str(i);
    extension1=str2double(extension);
    filename=strcat(baseFileName,extension,'.csv');
    name='currentfolder/filename';
    [y,TXT,RAW]=xlsread(filename);
    sensor=y(:,5);
    text(i,1)=TXT(2,5);
    for j=1:23
        num=selected_channels(1,j);
        k=Ns*(num-1)+1;
        p=k+255;
        new_data(:,j)=sensor(k:p,1);
        k=0;
        p=0;
    end
    for j=1:23
        S=new_data(:,j);
        waveletFunction = 'db8';
        [C,L] = wavedec(S,5,waveletFunction);
        D1 = wrcoef('d',C,L,waveletFunction,1); %NOISY
         for k=2:5   %2 3 4 5
         signal = wrcoef('d',C,L,waveletFunction,k); %D2--gamma, %D3--beta, %D4--alpha, %D5--theta
         Wave=stat_coeff_of_EEG(signal);
         sample_entrop=samp_entrop(signal);
         [LLE lambda]=lyaprosen(signal*100,0,0);
         wave=horzcat(wave,Wave,sample_entrop,LLE);         
         end
     signal=wrcoef('a',C,L,waveletFunction,5); %A5--delta
     Wave1=stat_coeff_of_EEG(signal);
     sample_entrop1=samp_entrop(signal);
     [LLE1 lambda1]=lyaprosen(signal*100,0,0);
     wave=horzcat(wave,Wave1,sample_entrop1,LLE1);     
    end
    final_wave=vertcat(final_wave,wave);
     Wave=[];
     sample_entrop=0;
     wave=[];
end


% x2Range='1';
% x3Range='A';
% sheet=1;
% Filename='F:\AUD files for LSTM\MTech project final susma\statistical coeff\Train wave.xlsx';
% xlswrite(Filename,final_wave,sheet,x2Range);
%    
% filename='F:\AUD files for LSTM\MTech project final susma\statistical coeff\labels train wave.xlsx';
% xlswrite(filename,text,sheet,x3Range);