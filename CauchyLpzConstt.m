function LpzH=CauchyLpzConstt(LPZ_TYPE,W,Q)
n=size(Q,2);
r=size(W,2);
switch upper(LPZ_TYPE),
    case 'PLAIN',   % Exact Lipschitz constants
        LpzH=ones(1,n);
        for i=1:n,
            LpzH(i)=norm(W'*((Q(:,i)*ones(1,r)).*W));
        end
    case 'CCOMP',   % Exact Lipschitz constants
        LpzH=CauchyLpzCC(W,Q);
    case 'RELAX',   % Inexact Lipschitz constants
        LpzH=norm(W')*sqrt(sum(Q.^2))*max(sqrt(sum(W.^2,2)));
    otherwise,
        error(['Unrecognized Lipschitz constant type: ',LPZ_TYPE]);
end
return;