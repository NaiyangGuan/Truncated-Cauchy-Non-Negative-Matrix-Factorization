function [H,Suppt,obj]=CauchyNLS(W,V,option,tol,verbose)

% Cauchy Regression by Nesterov's Method plus Half-Quadratic Programming.
% Written by Naiyang Guan (ny.guan@gmail.com).

[m,n]=size(V);
r=size(W,2);

MIN_ITER=20;    % minimum number of outer iterations
MAX_ITER=500;   % maximum number of outer iterations
ALG_TYPE='ogm';        % algorithm's type, default is OGM
WEI_TYPE='plain';      % weighting's type, default is PLAIN
LPZ_TYPE='plain';      % Lipschitz constant's type, default is PLAIN
SCL_TYPE='nagy';      % algorithm's type of scale estimation
gamma=0.1;      % scale parameter
H=rand(r,n);
% Algorithm options
if ~isempty(option),
    if isfield(option, 'min_iter'), MIN_ITER=option.min_iter;   end
    if isfield(option, 'max_iter'), MAX_ITER=option.max_iter;   end
    if isfield(option, 'alg_type'), ALG_TYPE=option.alg_type;   end
    if isfield(option, 'wei_type'), WEI_TYPE=option.wei_type;   end
    if isfield(option, 'img_info'), IMG_INFO=option.img_info;   end
    if isfield(option, 'lpz_type'), LPZ_TYPE=option.lpz_type;   end
    if isfield(option, 'gamma'),    gamma=option.gamma; end
    if isfield(option, 'h_init'),   H=option.h_init;    end
end
% Initialize accordingly
switch upper(WEI_TYPE),
    case {'ROBUSTG','ROBUSTL','MEDFILT'},
        E=V-W*H;
        delta=sum(sum(E.^2))/(2*m*n);
        Q=exp(-(E.^2)/(2*delta));
        H=H.*(W'*(Q.*V))./(W'*(Q.*(W*H))+eps);
    case 'PLAIN',
        Z=W*H;
        s=max(Z(:))/median(V(:));
        H=H/sqrt(s);
    case 'MRF',
        E=V-W*H;
        delta=sum(sum(E.^2))/(2*m*n);
        Q=exp(-(E.^2)/(2*delta));
        H=H.*(W'*(Q.*V))./(W'*(Q.*(W*H))+eps);
        height=IMG_INFO(1);
        width=IMG_INFO(2);
        NS=edges4connected(height,width);
        tau=0.17;
        lambda=3;
    otherwise,
        error(['Unrecognized weighting type: ',WET_TYPE]);
end
% Algorithm settings
switch upper(ALG_TYPE),
    case 'OGM',
        tolH=max(tol,1e-3);
    case 'BCD',
        H0=H;
        alpha0=1;
    otherwise,
        error(['Unrecognized algorithm type: ',ALG_TYPE]);
end
iterH=1;
E=abs(V-W*H);
% scale parameter
if (gamma<0),
    gamma=CauchySCL(0.1,E,SCL_TYPE);
else
    SCL_TYPE='fixed';
end
Z=(E/gamma).^2+1;
obj=sum(log(Z(:)))+m*n*log(gamma);
if verbose,
    fprintf('********** Cauchy NLS **********\n');
    fprintf('iter=%d, iterH=%d, objective function=%f, gamma=%f.\n',0,0,obj(0+1),gamma);
end
for iter=1:MAX_ITER,
    % Update Weights
    switch upper(WEI_TYPE),
        case 'PLAIN',
            Q=1./Z;
        case 'ROBUSTG',
            Q=1./Z;
            IX=CauchyOutlIndex(E,'global');
            Q(IX)=0;
        case 'ROBUSTL',
            Q=1./Z;
            IX=CauchyOutlIndex(E,'local');
            Q(IX)=0;
            Q=CauchyMedFilter(Q,IMG_INFO(1),IMG_INFO(2));
        case 'SHRINK',
            Q=1./Z;
            Q=(exp(Q*iter)-1);
        case 'MEDFILT',
            Q=1./Z;
            Q=MedFilter(Q,img_info(1),img_info(2));
        case 'MRF',
            Q=1./Z;
            IX=true(m,n);
            for j=1:n,
                s=MRF_ESuppt(E(:,j),IMG_INFO,NS,tau,lambda);
                IX(:,j)=(s==1);
            end
            Q(IX)=0;
        otherwise,
            error(['Unrecognized weighting type: ',WEI_TYPE]);
    end
    % Update H
    switch upper(ALG_TYPE),
        case 'OGM',
            [H,iterH]=CauchyOGM(V,W,H,Q,10,tolH,300,LPZ_TYPE);
            if iterH<=10,
                tolH=tolH/10;
            end
        case 'BCD',
            GradH0=W'*(Q.*(W*H0-V));    % Gradient at search point
            LpzH=CauchyLpzConstt(LPZ_TYPE,W,Q);
            H1=max(0,H0-GradH0./(ones(r,1)*LpzH));  % New 'H' solution
            alpha1=(1+sqrt(4*alpha0^2+1))/2;
            H0=H1+(alpha0-1)*(H1-H)/alpha1;
            H=H1;                       % Update for Next Recursion
    end
    E=abs(V-W*H);
    switch upper(SCL_TYPE),
        case {'NAGY','NEWTON'},
            gamma=CauchySCL(gamma,E,SCL_TYPE);
        case 'FIXED',
    end
    Z=(E/gamma).^2+1;
    % Check for stopping
    obj(iter+1)=sum(log(Z(:)))+m*n*log(gamma);
    if verbose,
        fprintf('iter=%d, iterH=%d, objective function=%f, gamma=%f.\n',iter,iterH,obj(iter+1),gamma);
    end    
    stop=abs(obj(end-1)-obj(end))/abs(obj(1)-obj(end));
    if stop<=tol && iter>=MIN_ITER,
        break;
    end
end
% Support of error, 1 for clean entry, and 0 for dirty entry
switch upper(WEI_TYPE),
    case {'ROBUSTG', 'ROBUSTL'},
        Suppt=~IX;
    otherwise,
        Suppt=true(m,n);
end
return;