

wave{1,1}='Gamma';
wave{1,2}='Beta';
wave{1,3}='Alpha';
wave{1,4}='Theta';
wave{1,5}='Delta';

param{1,1}='_power';
param{1,2}='_mean';

param{1,3}='_max-amp';
param{1,4}='_min-amp';


m=1;

    for j=1:5
        for k=1:4
            name{1,m}=strcat(wave{1,j},param{1,k});
            m=m+1;
        end
    end

x3Range='A1';
sheet=1;

Filename='F:\AUD files for LSTM\MTech project final susma\statistical coeff\fft\Train fft.xlsx';
xlswrite(Filename,name,sheet,x3Range);