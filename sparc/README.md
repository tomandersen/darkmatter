# Dark Matter Rotation Curve Simulator

##SPARC plan 
Read the file MassModels_Lelli2016c.mrt - generate the same graphs as I have (saved for reference in) MassModels_LTG_png folder. (data from [https://astroweb.case.edu/SPARC/](https://astroweb.case.edu/SPARC/)).  Here is a data sample. 

IE: for each galaxy, the data looks like this 
    
 NAME       distance R      Vobs e_Vobs v_gas  v_disk v_bulge SBDisk SBbul:

CamB          3.36   0.16   1.99  1.50   1.86   3.75   0.00   30.32     0.00 
CamB          3.36   0.41   4.84  1.50   4.24   9.47   0.00   23.77     0.00 
CamB          3.36   0.57   6.79  1.50   5.61  11.76   0.00   15.87     0.00 
CamB          3.36   0.73   8.87  1.50   6.77  13.72   0.00   12.40     0.00 
CamB          3.36   0.90  10.90  1.50   7.77  14.80   0.00    9.63     0.00 
CamB          3.36   1.06  12.90  1.50   8.44  15.24   0.00    5.86     0.00 
CamB          3.36   1.22  14.70  1.50   8.64  15.11   0.00    5.19     0.00 
CamB          3.36   1.47  16.80  1.50   8.08  15.90   0.00    3.02     0.00 
CamB          3.36   1.79  20.10  1.50   6.91  14.91   0.00    0.88     0.00 

For all the lines that have the same NAME, that is one galaxy, so you will plot a separate graph for each galaxy.
(You will have many graphs, as there are many galaxies in the file.)
For each galaxy - y axis will be v and x axis will be R, and you want to match the format of the graphs in the MassModels_LTG_png folder, roughly. Pick the y axis range to always start at zero and go up to a value larger than the max value of v for that galaxy. Same for x (0 to max(R) + 5 or so).

The distance column is ignored.
The R column is the radius of the point, and each v column is the observed velocity. For Vobs there are error bars, so you will plot the error bars as well (e_Vobs).
gas is v_gas (no error bars) GREEN dotted
disk is v_disk (no error bars) red dashed
bulge is v_bulge (no error bars) orange dashed
total = v_gas + v_disk + v_bulge (no error bars)blue
total is v_obs + error bars  BLUE solid

COnsidering the data, that will be making a loop to create about 175 PNGs in a folder, call the path MassModels_LTG_png/initial .

Once that is done, then I want to add a new line to each graph Outut graphs go in the curves folder, with subfolders in the curves folder for each set of settings i try. 

The new line on the graphs will be a function of the gas, stars, etc from the MassModels_Lelli2016c.mrt data. As an inital POC, you can add a new line called 'dm' by doubling the gas velocity (v_gas * 2).

