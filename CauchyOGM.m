function [H,iterH]=CauchyOGM(V,W,H0,Q,iterMin,tolH,iterMax,LPZ_TYPE)
[r,n]=size(H0);
L=CauchyLpzConstt(LPZ_TYPE,W,Q);
Y=H0;   % Search Point
Grad=W'*(Q.*(W*Y-V));   % Gradient at Y
alpha0=1;
init_pgn=GetStopCriterion(1,Y,Grad);
for k=0:iterMax-1,
    % Gradient Descent and Update
    H1=max(0,Y-Grad./(ones(r,1)*L));
    alpha1=(1+sqrt(4*alpha0^2+1))/2;
    Y=H1+(alpha0-1)*(H1-H0)/alpha1;
    % Update for Next Recursion
    H0=H1;
    alpha0=alpha1;
    Grad=W'*(Q.*(W*Y-V));
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