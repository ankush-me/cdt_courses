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
d  = size(A,1);


%% Q1 : solve for P : P>0, A'P + PA < 0
disp(['1 : Lyapunov using LMI:'])
cvx_begin sdp quiet
	variable P_psd(d,d)
	-A'*P_psd - P_psd*A - ee*eye(d) == semidefinite(d)
	P_psd - ee*eye(d) == semidefinite(d);
cvx_end
P_pd = P_psd + ee*eye(d)
disp(['eigenvalues of P = ', num2str(eig(P_pd)')])
disp(['eigenvalues of A^TP+PA = ', num2str(eig(A'*P_pd+P_pd*A)')])
disp(' ')
disp('press any key to continue ...');
disp(' ')
pause();


%% 2(a) : Diagonal Lyapunov:
disp(['2(a) : Diagonal lyapunov:'])
cvx_begin sdp quiet
	variable p(d)
	-A'*(diag(p)) - (diag(p))*A -ee*eye(d) == semidefinite(d);
	p >= ee;
cvx_end
if ~strcmp(cvx_status, 'Infeasible')
	P_pd = diag(p) + ee*eye(d)
	disp(['   eigenvalues of P = ', num2str(eig(P_pd)')])
	disp(['   eigenvalues of A^TP+PA = ', num2str(eig(A'*P_pd+P_pd*A)')])
else
	disp(['   > this SDP is INFEASIBLE. No diagonal P exists.'])
end
disp(' ')
disp('press any key to continue ...');
disp(' ')
pause();


%% 2(a) : Diagonal Lyapunov:
disp(['2(b) : Sparse Lyapunov:'])
cvx_begin sdp quiet
	variable P_psd(d,d)
	minimize(norm(vec(P_psd), 1))
	-A'*P_psd - P_psd*A - ee*eye(d) == semidefinite(d)
	P_psd - ee*eye(d) == semidefinite(d);
cvx_end
disp(['BEFORE THRESHOLDING : '])
P_pd = P_psd + ee*eye(d)
disp(['    eigenvalues of P = ', num2str(eig(P_pd)')])
disp(['    eigenvalues of A^TP+PA before thersholding = ', num2str(eig(A'*P_pd+P_pd*A)')])
disp(['AFTER THRESHOLDING (epsilon = 1e-3) : '])
P_pd(abs(P_pd)<1e-3) = 0.0
disp(['    eigenvalues of P = ', num2str(eig(P_pd)')])
disp(['    eigenvalues of A^TP+PA = ', num2str(eig(A'*P_pd+P_pd*A)')])
disp(' ')
disp('press any key to continue ...');
disp(' ');
pause();

% 3(a) : Stabilizable:
disp(['3(a) : Stabilizable:'])
cvx_begin sdp quiet
	variable Q_psd(d,d)
	variable s
	s >= eps
	Q_psd - ee*eye(d) == semidefinite(d);
	-A'*Q_psd - Q_psd*A - s*B*B' - ee*eye(d) == semidefinite(d)
cvx_end
Q_pd = Q_psd + ee*eye(d);
K = -s/2*B'*inv(Q_psd)
disp(['checking if closed-loop is stable <=> A_cl is Hurwitz:'])
A_cl = A + B*K;
disp(['eigenvalues of A_cl = ', num2str(eig(A_cl)')])
if all(real(eig(A_cl)) < 0)
	disp(' >> closed-loop system is stable')
else
	disp(' >> closed-loop system is NOT stable')
end
disp(' ')
disp('press any key to continue ...');
disp(' ')
pause();

%% 3(v) : Stabilizable:
disp(['3(b) : Stabilizable (alternate method) :'])
disp(['    >> LMI : AQ + QA'' + BY + Y''B'' < 0 s.t. Q>0 | Y==KQ'])
cvx_begin sdp quiet
	variable Q_psd(d,d)
	variable Y(size(B,2),d)
	Q_psd - ee*eye(d) == semidefinite(d);
	-A*Q_psd - Q_psd*A' - B*Y - Y'*B' - ee*eye(d) == semidefinite(d)
cvx_end
Q_pd = Q_psd + ee*eye(d);
K = Y*inv(Q_pd)

disp(['checking if closed-loop is stable <=> A_cl is Hurwitz:'])
A_cl = A + B*K;
disp(['eigenvalues of A_cl = ', num2str(eig(A_cl)')])
if all(real(eig(A_cl)) < 0)
	disp(' >> closed-loop system is stable')
else
	disp(' >> closed-loop system is NOT stable')
end
disp(' ')



