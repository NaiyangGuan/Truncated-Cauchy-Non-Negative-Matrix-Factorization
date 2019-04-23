function [W,H,HIS]=CauchyNMF_basis(V,r,varargin)

% Cauchy Nonnegative Matrix Factorization via Nesterov Based Half-Quadratic Programming.

% The model is V \approx WH, where V, W, and H are defined as follows:
% V (m x n): data matrix including n samples in m-dimensional space;
% W (m x r): basis matrix including r bases in m-dimensional space;
% H (r x n): coefficients matrix including n encodings in r-dimensional space.

% Written by Naiyang Guan (ny.guan@gmail.com).
% Copyright @ 2013-2023 by Naiyang Guan and Dacheng Tao.
% Last Modified at
%   Sep. 06 2014;
%   Nov. 13 2014;
%   Aug. 07 2015;
%   Aug. 12 2015;
%   Jan. 06 2016;
%   Jan. 16 2017.

% <Inputs>
%        V : Input data matrix (m x n), should be scaled to [0,1] in the
%        default sense
%        r : Target low-rank
%
%        (Below are optional arguments: can be set by providing name-value
%        pairs).
%        MAX_ITER : Maximum number of iterations. Default is 1,000.
%        MIN_ITER : Minimum number of iterations. Default is 10.
%        MAX_TIME : Maximum amount of time in seconds. Default is 10,000.
%        ALG_TYPE : Type of algorithms.
%           'OGM' for Nesterov's optimal gradient method,
%           'MUR' for multiplicative update rule,
%           'HALS' for weighted HALS algorithm,
%           'BCD' for linear-proximal block coordinate descent.
%        WEI_TYPE : Type of weighting.
%           'plain' for original data-adapative weighting,
%           'tgv' for robustly trimmed weighting in global viewpoint,
%           'tlv' for robustly trimmed weighting in local viewpoint,
%           'mrf' for trimmed weighting by detecting supports by MRF,
%           'med' for trimmed weighting with median filtering.
%        LPZ_TYPE : Type of Lipschitz constant.
%           'plain' for calculating by Matlab,
%           'ccomp' for calculating by MEX-C and OpenMP,
%           'relax' for approximation.
%        IMG_INFO : Struct format, contains two records, i.e., height and
%        width, formatted by [height,width].
%        SCALE : Scale parameter. Default is 0.1. If gamma < 0, automatic.
%           -1 : Nagy algorithm for scale estimation,
%           -2 : Newton algorithm for scale estimation.
%        W_INIT : (m x r) Initial value for W.
%        H_INIT : (r x n) Initial value for H.
%        TOL : Stopping tolerance, [outer one, inner one]. Default is [1e-4,1e-2].
%           If you want to obtain a more accurate solution, decrease TOL and increase MAX_TIME simultanuously.
%        VERBOSE : 0 (default) - No debugging information.
%                  1 (debugging purpose) - History of computation is printed on screen.
%                  2 - History of basis.
% <Outputs>
%        W : Obtained basis matrix (m x r),
%        H : Obtained coefficients matrix (r x n),
%        HIS : (debugging purpose) History of computation.
%
% <Usage Examples>
%        >>A=rand(100);
%        >>CauchyNMF(A,10);
%        >>CauchyNMF(A,20,'verbose',1);
%        >>CauchyNMF(A,30,'verbose',2,'w_init',rand(100,30));
%        >>CauchyNMF(A,5,'verbose',2,'tol',1e-5);

[m,n]=size(V);
if ~isa(V,'double'),    V=double(V);    end
if ~exist('V','var'),    error('please input the sample matrix.\n');    end
if ~exist('r','var'),    error('please input the low rank.\n'); end
% if n<=r || m<=r,    error('Too high low-rank (n<=r or m<=r).\n');    end

% Default setting
MIN_ITER=20;    % minimum number of outer iterations
MAX_ITER=500;   % maximum number of outer iterations
INN_MIN=20;    % minimum number of inner iterations
INN_MAX=300;    % maximum number of inner iterations
MAX_TIME=10000; % maximum running time
ALG_TYPE='ogm';        % algorithm's type, default is OGM
WEI_TYPE='plain';      % weighting's type, default is PLAIN
LPZ_TYPE='plain';      % Lipschitz constant's type, default is PLAIN
SCL_TYPE='plain';      % algorithm's type of scale estimation
gamma=0.1;             % scale parameter
tol=[1e-4,1e-2];       % tolerance of outer and inner loops
verbose=0;
img_info=[sqrt(m),sqrt(m)];     % information [height, width] for images
W0=rand(m,r);
H0=rand(r,n);

% Read optional parameters
if (rem(length(varargin),2)==1),
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1),
        switch upper(varargin{i}),
            case 'MAX_ITER',    MAX_ITER=varargin{i+1};
            case 'MIN_ITER',    MIN_ITER=varargin{i+1};
            case 'MAX_TIME',    MAX_TIME=varargin{i+1};
            case 'ALG_TYPE',    ALG_TYPE=varargin{i+1};
            case 'WEI_TYPE',    WEI_TYPE=varargin{i+1};
            case 'LPZ_TYPE',    LPZ_TYPE=varargin{i+1};
            case 'IMG_INFO',    img_info=varargin{i+1};
            case 'SCALE',       gamma=varargin{i+1};
            case 'W_INIT',      W0=varargin{i+1};
            case 'H_INIT',      H0=varargin{i+1};
            case 'TOL',         tol=varargin{i+1};
            case 'VERBOSE',     verbose=varargin{i+1};
            otherwise
                error(['Unrecognized option: ',varargin{i}]);
        end
    end
end

% Initialization
switch upper(WEI_TYPE),
    case 'PLAIN',
        Z=W0*H0;
        s=max(Z(:))/max(median(V(:)),min(Z(:)));
        W0=W0/sqrt(s);
        H0=H0/sqrt(s);
    case {'TGV', 'TLV', 'MED'},
        R=V-W0*H0;
        delta=sum(sum(R.^2))/(2*m*n);
        Q=exp(-(R.^2)/(2*delta));
        W0=W0.*((Q.*V)*H0')./((Q.*(W0*H0))*H0'+eps);
        H0=H0.*(W0'*(Q.*V))./(W0'*(Q.*(W0*H0))+eps);
    case 'MRF',
        R=V-W0*H0;
        delta=sum(sum(R.^2))/(2*m*n);
        Q=exp(-(R.^2)/(2*delta));
        W0=W0.*((Q.*V)*H0')./((Q.*(W0*H0))*H0'+eps);
        H0=H0.*(W0'*(Q.*V))./(W0'*(Q.*(W0*H0))+eps);
        height=img_info(1);
        width=img_info(2);
        NS=edges4connected(height,width);
        tau=0.17;
        lambda=3;
    otherwise,
        error(['Unknown weighting type: ',WEI_TYPE]);
end
Z=W0*H0;

% Algorithmic Settings
switch upper(ALG_TYPE),
    case 'OGM',
        tolH=max(tol(2),1e-3);
        tolW=max(tol(2),1e-3);
    case 'BCD',
        alpha0=1;
end
iterH=0;
iterW=0;
W=W0;       % In case of LPRBCD, 'W' is current solution, 'W0' stores search point.
H=H0;       % In case of LPRBCD, 'H' is current solution, 'H0' stores search point.
R=(V-W*H);          % Residual Error
if (gamma<0),
    switch gamma,
        case -1,
            SCL_TYPE='nagy';
        case -2,
            SCL_TYPE='newton';
        otherwise,
            error(['Unknown gamma value: ',int2str(gamma)]);
    end
end
switch upper(SCL_TYPE),
    case {'NAGY','NEWTON'},
        gamma=CauchySCL(10,R,SCL_TYPE);
    case 'PLAIN',
    otherwise,
        error(['Unrecognized scale type: ',SCL_TYPE]);
end
Z=(R/gamma).^2+1;
HIS.obj=sum(log(Z(:)))+m*n*log(gamma);
HIS.sec=0;
HIS.gamma=gamma;
if verbose,
    fprintf('********** Cauchy NMF **********\n');
    fprintf('%d: iterH=%d, iterW=%d, scale=%.3f, obj=%f.\n',0,iterH,iterW,gamma,HIS.obj);
    if verbose==2,
        H=H.*(sum(W)'*ones(1,n));
        W=W./(ones(m,1)*sum(W));
        HIS.basis=num2cell(W0);
        HIS.basis={HIS.basis,num2cell(W0)};
    end
end
tic;
for iter=1:MAX_ITER,
    % Update Weights
    switch upper(WEI_TYPE),
        case 'PLAIN',
            Q=1./Z;
        case 'TGV', % For small samples
            Q=1./Z;
            IX=CauchyOutlIndex(abs(R),'global');
            Q(IX)=0;
        case 'TLV', % For large samples
            Q=1./Z;
            IX=CauchyOutlIndex(abs(R),'local');
            Q(IX)=0;
        case 'MED',
            Q=1./Z;
            Q=CauchyMedFilter(Q,img_info(1),img_info(2));
        case 'MRF',
            Q=1./Z;
            IX=true(m,n);
            for j=1:n,
                s=MRF_ESuppt(R(:,j),img_info,NS,tau,lambda);
                IX(:,j)=(s==1);
            end
            Q(IX)=0;
        otherwise,
            error(['Unrecognized weighting type: ',WET_TYPE]);
    end
    % Update H
    switch upper(ALG_TYPE),
        case 'OGM',
            [H,iterH]=CauchyOGM(V,W,H,Q,INN_MIN,tolH,INN_MAX,LPZ_TYPE);
            if iterH<=10,
                tolH=tolH/10;
            end
            R=(V-W*H);
        case 'MUR',
            H=H.*(W'*(Q.*V))./(W'*(Q.*(W*H))+eps);
            R=(V-W*H);
        case 'HALS',
            for k=1:r,
                R=R+W(:,k)*H(k,:);
                Wk=W(:,k)*ones(1,n);    % Update row of H
                H(k,:)=max(0,sum(Q.*R.*Wk)./(sum(Q.*Wk.*Wk)+eps));
                Hk=ones(m,1)*H(k,:);    % Update column of W
                W(:,k)=max(0,sum(Q.*R.*Hk,2)./(sum(Q.*Hk.*Hk,2)+eps));
                R=R-W(:,k)*H(k,:);
            end
        case 'BCD',
            GradH0=W'*(Q.*(W*H0-V));     % Gradient at search point
            LpzH=CauchyLpzConstt(LPZ_TYPE,W,Q);
            H1=max(0,H0-GradH0./(ones(r,1)*LpzH));  % New 'H' solution
            alpha1=(1+sqrt(4*alpha0^2+1))/2;
            H0=H1+(alpha0-1)*(H1-H)/alpha1;
            H=H1;                       % Update for Next Recursion
            R=(V-W*H);
        otherwise,
            error(['Unrecognized algorithm type: ',ALG_TYPE]);
    end
    % Update W
    switch upper(ALG_TYPE),
        case 'OGM',
            [W,iterW]=CauchyOGM(V',H',W',Q',INN_MIN,tolW,INN_MAX,LPZ_TYPE);
            W=W';
            if iterW<=10,
                tolW=tolW/10;
            end
            R=(V-W*H);
        case 'MUR',
            W=W.*((Q.*V)*H')./((Q.*(W*H))*H'+eps);
            R=(V-W*H);
        case 'HALS',
        case 'BCD',
            GradW0=(Q.*(W0*H-V))*H';     % Gradient at search point            
            LpzW=CauchyLpzConstt(LPZ_TYPE,H',Q');
            LpzW=LpzW';
            W1=max(0,W0-GradW0./(LpzW*ones(1,r)));  % New 'W' solution
            W0=W1+(alpha0-1)*(W1-W)/alpha1;
            W=W1;                       % Update for Next Recursion
            alpha0=alpha1;
            R=(V-W*H);
        otherwise,
            error(['Unrecognized algorithm type: ',ALG_TYPE]);
    end
    % Normalization
    if verbose==2,
        H=H.*(sum(W)'*ones(1,n));
        W=W./(ones(m,1)*sum(W));
    else
        H=H.*(max(W)'*ones(1,n));
        W=W./(ones(m,1)*max(W));
    end
    switch upper(SCL_TYPE),
        case {'NAGY','NEWTON'},
            gamma=CauchySCL(gamma,R,SCL_TYPE);
        case 'PLAIN',
        otherwise,
            error(['Unrecognized scale type: ',SCL_TYPE]);
    end
    Z=(R/gamma).^2+1;
    % Objective Function
    HIS.obj(iter+1)=sum(log(Z(:)))+m*n*log(gamma);
    HIS.sec(iter+1)=toc;
    HIS.gamma(iter+1)=gamma;
    if (rem(iter,10)==0) && verbose,
        fprintf('%d: iterH=%d, iterW=%d, scale=%.3f, obj=%f.\n',iter,iterH,iterW,gamma,HIS.obj(end));
    end
    if verbose==2,
        HIS.basis{iter+1}=num2cell(W);
    end
    stop=abs(HIS.obj(end-1)-HIS.obj(end))/abs(HIS.obj(1)-HIS.obj(end));
    if (stop<=tol(1) && iter>=MIN_ITER) || (HIS.sec(end)>=MAX_TIME),
        break;
    end
end
% Support for residual error, 1 for clean entry, and 0 for dirty entry
switch upper(WEI_TYPE),
    case {'TGV', 'TLV'},
        HIS.suppt=~IX;
    otherwise,
        HIS.suppt=true(m,n);
end
return;