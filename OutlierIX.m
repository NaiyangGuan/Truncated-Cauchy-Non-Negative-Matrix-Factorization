function IX=OutlierIX(Z)
x=Z(Z<=median(Z(:)));
IX=(abs(Z-mean(x))>=3*std(x));
return;