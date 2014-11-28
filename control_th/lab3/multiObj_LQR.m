%% Multi Objective LQR Controller Design:
%% ======================================

close all;
clear; clc;

A = [-0.003    0.039     0.000   -0.322;
	 -0.065   -0.319     7.740    0.000;
	  0.020   -0.101    -0.429    0.000;
	  0.000    0.000     1.000    0.000];

B = [ 0.01    1.000;
     -0.18   -0.040;
     -1.16    0.598;
      0.00    0.000 ];

C = [0 1 0 0; 0 0 1 0];
D = zeros(4,2);

ee = 1e-4;
nx = size(A,1);
nu = size(B,2);

Q = eye(nx);
R = 0.1*eye(nu);

Qsq = sqrtm(Q);
Rsq = sqrtm(R);

%% |Im(lambda)| >= 2.5 constraint params:
beta = 2.5;
L1 = [-2*beta 0;
       0    -2*beta];
M1 = [0 1; -1 0];

%% within cone of 30 degrees :
th = pi/180*30;
L2 = zeros(2);
M2 = [sin(th) cos(th); -cos(th) sin(th)];

cvx_begin sdp quiet
	variable S(nx,nx)
	variable Y(nu,nx)

	maximize trace(S);

	-[ A*S+S*A'+Y'*B'+B*Y     S*Qsq        Y'*Rsq;
	       Qsq*S          -eye(nx)     zeros(nx,nu);
	       Rsq*Y        zeros(nu,nx)      -eye(nu)] == semidefinite(2*nx+nu);

	S - ee*eye(nx) == semidefinite(nx);

	%% |Im(lambda)| >= 2.5
	-kron(L1,S) - kron(M1, A*S+B*Y) - kron(M1', S*A'+Y'*B') - ee*eye(2*nx) == semidefinite(2*nx);

	%% cone of 30 degrees:
	-kron(L2,S) - kron(M2, A*S+B*Y) - kron(M2', S*A'+Y'*B') - ee*eye(2*nx) == semidefinite(2*nx);

cvx_end
K = -Y*inv(S);
Acl = A - B*K;
ev = eig(Acl);
disp(['Eigenvalues of closed-loop system : ' num2str(eig(Acl)')])

%% check for the 10% variation in 7.74:
Am = A;
Am(2,3) = 0.9*A(2,3);
Ap = A;
Ap(2,3) = 1.1*A(2,3);
Am_cl = Am-B*K;
Ap_cl = Ap-B*K;
disp(['Eigenvalues of [A - BK] = ', num2str(eig(Acl)')])
if all(real(eig(Acl)) < 0)
	disp('   >> closed-loop system is stable')
else   
	disp('   >> closed-loop system is NOT stable')
end
disp(' ')

hold on;
fhigh = @(x)min(-tan(th)*x, beta);
flow = @(x)max(tan(th)*x, -beta);
xs = -10:0.1:0;
ylow = flow(xs);
yhigh = fhigh(xs);
ylabel('imag(\lambda)');
xlabel('real(\lambda)');
fill([xs,xs(end:-1:1)], [yhigh,ylow(end:-1:1)], 'b', 'faceAlpha', 0.1);
fplot(fhigh, [-10 0]);
fplot(flow, [-10 0]);
axis([-10, 5, -4, 4]);
axis equal;
scatter(real(ev), imag(ev), 50, 'black', 'filled');

disp(['Eigenvalues of [A(-) - BK] = ', num2str(eig(Am_cl)')])
if all(real(eig(Am_cl)) < 0)
	disp('   >> 90% of 7.74 closed-loop system is stable')
else   
	disp('   >> 90% of 7.74 closed-loop system is NOT stable')
end
disp(' ')

disp(['Eigenvalues of [A(+) - BK] = ', num2str(eig(Ap_cl)')])
if all(real(eig(Ap_cl)) < 0)
	disp('   >> 110% of 7.74 closed-loop system is stable')
else
	disp('   >> 110% of 7.74 closed-loop system is NOT stable')
end



