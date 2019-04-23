function Index=CauchyOutlIndex(R,view)
    switch upper(view),
        case 'GLOBAL',
            x=R(R<=median(R(:)));
            Index=(abs(R-mean(x))>=3*std(x));
        case 'LOCAL',
            n=size(R,2);    % Number of samples
            medn=median(R,2)*ones(1,n); % Median of each dimension
            id=(R<=medn);   % Indicator of entries under median
            nn=sum(id,2);   % Number of entries under median
            mn=(sum(R.*id,2)./nn)*ones(1,n);   % Mean of each dimension
            stdn=sqrt((sum((((R-mn).*id).^2),2))./(nn-1))*ones(1,n);   % Standard deviation of each dimension
            Index=(abs(R-mn)>=3*stdn);  % Outlier indicator by three-sigma theory
        otherwise,
            error(['Unknown viewpoint: ',view]);
    end
return;