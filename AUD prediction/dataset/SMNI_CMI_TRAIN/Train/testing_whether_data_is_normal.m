Ns=256;
sensor=[];
numexcelfiles=1;  %file number of the last excel sheet is Data468.csv
text=string.empty;
x2Range='2';
x3Range='A';
sheet=1;
base='Data__';
Name= 'F:\AUD files for LSTM\MTech project final susma\real time series\Train\';
currentfolder=cd;
for i=1:numexcelfiles    
    baseFileName='Data';
    extension=num2str(i);
    extension1=str2double(extension);
    filename=strcat(baseFileName,extension,'.csv');
    name='currentfolder/filename';
    avg_raw=zeros(Ns,1);   
    [y,TXT,RAW]=xlsread(filename);  
    boody_boo(i,1)=TXT(2,5);
    sensor=y(:,5);
    
    

    
%      extension=num2str(i);
%      Filename=strcat(base,extension,'.xlsx');
%      file=strcat(Name,Filename);
%      xlswrite(file,wave,sheet,x2Range); 
end
        
%  filename='F:\AUD files for LSTM\MTech project final susma\real time series\Train\labels train.xlsx';
%  xlswrite(filename,boody_boo,sheet,x3Range);






