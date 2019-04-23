function H=CauchyNNLS(W,V,gamma,tol)

n=size(V,2);
r=size(W,2);
H=rand(r,n);
tolH=max(tol,1e-2*ones(1,n));
iterH=zeros(1,n);
E=abs(V-W*H);
% scale parameter
if gamma<0,
    gamma=sqrt(sum(sum(E.^2))/(2*m*n));
end
Z=(E/gamma).^2+1;
obj=sum(log(Z(:)));
verbose=0;

if verbose,
    fprintf('********** CauchyNNLS **********\n');
end

for iter=1:1000,
    % Update H
    Q=1./Z;
    IX=OutlierIX(E);
    nn=sum(IX,2);
    IX(nn<=median(nn),:)=0;
    Q(IX)=0;

    for j=1:n,
        [Hj,iterH(j)]=OGM(V(:,j),W,H(:,j),Q(:,j),10,tolH(j),1000);
        H(:,j)=Hj;
        if iterH(j)<=10,
            tolH(j)=tolH(j)/10;
        end
    end
    E=abs(V-W*H);
    if gamma<0,
        gamma=sqrt(sum(sum(E.^2))/(2*m*n));
    end
    Z=(E/gamma).^2+1;
    
    obj(iter+1)=sum(log(Z(:)));
    if verbose,
        fprintf('iter=%d, objective function=%f, gamma=%f.\n',iter,obj(iter+1),gamma);
    end
    
    stop=abs(obj(end-1)-obj(end))/abs(obj(1)-obj(end));
    if stop<=tol && iter>=10,
        break;
    end
end

return;

function [H,iterH]=OGM(V,W,H0,Q,iterMin,tolH,iterMax)
WtW=W'*((Q*ones(1,size(W,2))).*W);
L=norm(WtW);
WtV=W'*((Q*ones(1,size(V,2))).*V);
Y=H0;   % Search Point
Grad=WtW*Y-WtV;   % Gradient at Y
alpha0=1;
init_pgn=GetStopCriterion(1,Y,Grad);
for k=0:iterMax-1,
    % Gradient Descent and Update
    H1=max(0,Y-Grad/L);
    alpha1=(1+sqrt(4*alpha0^2+1))/2;
    Y=H1+(alpha0-1)*(H1-H0)/alpha1;
    % Update for Next Recursion
    H0=H1;
    alpha0=alpha1;
    Grad=WtW*Y-WtV;
    % Stopping Criteria
    if k>=iterMin-1,
        % Lin's stopping condition
        pgn=GetStopCriterion(1,Y,Grad);
        if pgn<init_pgn*tolH,
            break;
        end
    end
end
H=max(0,Y);
iterH=k+1;
return;