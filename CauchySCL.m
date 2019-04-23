function delta=CauchySCL(delta0,E,scl_type)
% Scale estimator by the Nagy algorithm:
% F. Nagy, ¡°Parameter Estimation of the Cauchy Distribution in Information Theory Approach,¡±...
%   Journal of Universal Computer Science, vol. 12, no. 9, pp. 1332-1344, 2006.
% Input: delta0 is the initial value of delta, and E is the error,
% Output: delta is the obtained scale parameter of Cauchy distribution.
% Written by Naiyang Guan (ny.guan@gmail.com)
a_vec=E(:).^2;
N=length(a_vec);
verbose=0;
if verbose,
    fprintf('%s method starting,\t initial value=%f.\n',scl_type,x);
end
switch upper(scl_type),
    case 'NAGY',
        x=4*delta0^2;   % initial from 2*delta0
        for iter=1:100,
            div=N/sum(1./(1+a_vec/x))-1;
            if abs(div-1)<1e-10,
                break;
            else
                x=x*div;
            end
            if verbose,
                fprintf('iter=%d,\t x=%f.\n',iter,x);
            end
        end
    case 'NEWTON',
        x=eps;  % initial from zero
        for iter=1:100,
            fx=sum(a_vec./(a_vec+x))-N/2;
            fx_div=-sum(a_vec./((a_vec+x).^2));
            if abs(fx)<1e-10,
                break;
            else
                x=x-fx/fx_div;
            end
            if verbose,
                fprintf('iter=%d,\t x=%f,\t fx=%f,\t fx_div=%f.\n',iter,x,fx,fx_div);
            end
        end
   otherwise,
        error(['Unknown scale type: ',scl_type]);
end
delta=sqrt(x);
if verbose,
    fprintf('%s method succeeds,\t final x=%f,\t final delta=%f.\n',scl_type,x,delta);
end
return;