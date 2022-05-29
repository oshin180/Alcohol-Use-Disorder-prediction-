wave{1,1}='Gamma';
wave{1,2}='Beta';
wave{1,3}='Alpha';
wave{1,4}='Theta';
wave{1,5}='Delta';

param{1,1}='_power';
param{1,2}='_phase';

m=1;


    for j=1:5
        for k=1:2
            name{1,m}=strcat(wave{1,j},param{1,k});
            m=m+1;
        end
    end


x3Range='A1';
sheet=1;
base='Data__';
Name= 'F:\AUD files for LSTM\MTech project final susma\real time series\Test\';

for i=1:480  %number of files is 480
       extension=num2str(i);
       Filename=strcat(base,extension,'.xlsx');
       file=strcat(Name,Filename);
       xlswrite(file,name,sheet,x3Range);
end