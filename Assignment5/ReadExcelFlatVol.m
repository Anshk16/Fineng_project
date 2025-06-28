function [FlatVol, Strikes] = ReadExcelFlatVol(filename)

FlatVol=readmatrix(filename,'Range','B2:N14');
FlatVol=FlatVol*1e-2;
Strikes=readmatrix(filename,'Range','B1:N1');
Strikes=Strikes*1e-2;
end

