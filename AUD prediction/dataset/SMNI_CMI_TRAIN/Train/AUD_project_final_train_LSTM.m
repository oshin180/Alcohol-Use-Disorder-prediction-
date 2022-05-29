Ns=256;
k=Ns;
b=0;
n=1;
raw=[];
sensor=[];
numexcelfiles=468;  %file number of the last excel sheet is Data468.csv
text=string.empty;
l=0;
mm=1;
x2Range='2';
x3Range='A';
sheet=1;
base='Data__';
Name= 'F:\AUD files for LSTM\MTech project final susma\real time series\Train\';
hilsine=[];
power=[];
phase=[];
hil=[];
currentfolder=cd;
for i=1:numexcelfiles    

    baseFileName='Data';
    extension=num2str(i);
    extension1=str2double(extension);
    filename=strcat(baseFileName,extension,'.csv');
    name='currentfolder/filename';
    avg_raw=zeros(Ns,1);
    
    [y,TXT,RAW]=xlsread(filename);    
    sensor=y(:,5);
    m=length(y)/256; %64       
    boody_boo(i,1)=TXT(2,5);
    %average multichannel signal
    while k<=16384
     for j=l:k-1        %l=0:256-1                 
         n=mod(j,Ns)+1;
         avg_raw(n,1)=avg_raw(n,1)+sensor(j+1);         
     end
     l=k;
     k=k+Ns;        
    end
    
    avg_raw=avg_raw/m;
    %wavelet transform
    waveletFunction = 'db8';
    [C,L] = wavedec(avg_raw,5,waveletFunction);
    %Calculation the Details Vectors of every Band :
    D1 = wrcoef('d',C,L,waveletFunction,1); %NOISY
    
    for q=2:5   %2 3 4 5
        signal = wrcoef('d',C,L,waveletFunction,q); %D2--gamma, %D3--beta, %D4--alpha, %D5--theta
        S=hilbert(signal);
        wave(:,mm)=abs(S).^2;  %power
        mm=mm+1;
        wave(:,mm)=angle(S); %phase
        mm=mm+1;
    end
    signal=wrcoef('a',C,L,waveletFunction,5); %A5--delta
    S=hilbert(signal);
    wave(:,mm)=abs(S).^2;
    mm=mm+1;
    wave(:,mm)=angle(S);
    
     extension=num2str(i);
     Filename=strcat(base,extension,'.xlsx');
     file=strcat(Name,Filename);
     xlswrite(file,wave,sheet,x2Range); 
    
      mm=1;
      sensor=[];
      wave=[];
      l=0;
      k=Ns;
      n=0;
end
        
 filename='F:\AUD files for LSTM\MTech project final susma\real time series\Train\labels train.xlsx';
 xlswrite(filename,boody_boo,sheet,x3Range);






