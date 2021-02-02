function insolarpluto(m::Array{T,1},xout::Array{T,2},vout::Array{T,2},pair::Array{Int64,2}) where {T <: Real}

# global YEAR GNEWT

# Below data from Hairer pg. 13-14.  Convert time to years.

# Distance is in AU, mass in solar masses.  

fac = 1;

m = fac*[1.00000597682,0.000954786104043,0.000285583733151,
         0.0000437273164546,0.0000517759138449,6.58086572e-9];

m[1] = 1.00000;

n = length(m);

for i=1:n

    for j=1:n

#         Kepler solver group

        if i == 1 || j == 1

            pair[i,j] = 0;

            pair[j,i] = 0;

        else

            pair[i,j] = 1;

            pair[j,i] = 1;

        end

    end

end

xout = [-2.079997415328555E-04, 7.127853194812450E-03,-1.352450694676177E-05,
        -3.502576700516146E+00,-4.111754741095586E+00, 9.546978009906396E-02,
         9.075323061767737E+00,-3.443060862268533E+00,-3.008002403885198E-01,
         8.309900066449559E+00,-1.782348877489204E+01,-1.738826162402036E-01,
         1.147049510166812E+01,-2.790203169301273E+01, 3.102324955757055E-01,
        -1.553841709421204E+01,-2.440295115792555E+01, 7.105854443660053E+00]

xout = xout';

vout = [-6.227982601533108E-06, 2.641634501527718E-06, 1.564697381040213E-07,
         5.647185656190083E-03,-4.540768041260330E-03,-1.077099720398784E-04,
         1.677252499111402E-03, 5.205044577942047E-03,-1.577215030049337E-04,
         3.535508197097127E-03, 1.479452678720917E-03,-4.019422185567764E-05,
         2.882592399188369E-03, 1.211095412047072E-03,-9.118527716949448E-05,
         2.754640676017983E-03,-2.105690992946069E-03,-5.607958889969929E-04];

 vout = vout';

vout = vout*YEAR;
 
xout = xout[:,1:n];
vout = vout[:,1:n];
vcm = zeros(3,1);
xcm = zeros(3,1);
for j=1:n
    vcm[:] = vcm[:]+m[j]*vout[:,j];
    xcm[:] = xcm[:]+m[j]*xout[:,j];
end
vcm = vcm/sum(m);
xcm = xcm/sum(m);
# Adjust so CoM is stationary
for j=1:n
    vout[:,j] = vout[:,j]-vcm[:]; 
    xout[:,j] = xout[:,j]-xcm[:];
end
# Add CoM velocity
vcm = 100*vcm;
vcm = 0*vcm;
for j=1:n
    vout[:,j] = vout[:,j] + vcm[:]; 
end
return
end
