close all;
clear; clc;
disp_title = @(s) disp(colorize(s,'green', true, false));
disp_blue = @(s) disp(colorize(s,'blue', true, false));
disp_red = @(s) disp(colorize(s,'red', false, false));
disp_yellow = @(s) disp(colorize(s,'yellow', false, false));

A = [-0.003    0.039     0.000   -0.322;
	 -0.065   -0.319     7.740    0.000;
	  0.020   -0.101    -0.429    0.000;
	  0.000    0.000     1.000    0.000];

B = [ 0.01    1.000;
     -0.18   -0.040;
     -1.16    0.598;
      0.00    0.000 ];

C = [0 1 0 0; 0 0 1 0];
D = zeros(2,2);

ee = 1e-5;
d  = size(A,1);


%% Q1 : solve for P : P>0, A'P + PA < 0
disp_title(['1 : Lyapunov using LMI:']);
cvx_begin sdp quiet
	variable P_psd(d,d)
	-A'*P_psd - P_psd*A - ee*eye(d) == semidefinite(d)
	P_psd - ee*eye(d) == semidefinite(d);
cvx_end
P_pd = P_psd + ee*eye(d)
disp_blue(['eigenvalues of P = ', num2str(eig(P_pd)')])
disp_blue(['eigenvalues of A^TP+PA = ', num2str(eig(A'*P_pd+P_pd*A)')])
disp(' ')
disp('press any key to continue ...');
disp(' ')
pause();


%% 2(a) : Diagonal Lyapunov:
disp_title(['2(a) : Diagonal lyapunov:'])
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
	disp_red(['   > this SDP is INFEASIBLE. No diagonal P exists.'])
end
disp(' ')
disp('press any key to continue ...');
disp(' ')
pause();


%% 2(a) : Diagonal Lyapunov:
disp_title(['2(b) : Sparse Lyapunov:'])
cvx_begin sdp quiet
	variable P_psd(d,d)
	minimize(norm(vec(P_psd), 1))
	-A'*P_psd - P_psd*A - ee*eye(d) == semidefinite(d)
	P_psd - ee*eye(d) == semidefinite(d);
cvx_end
disp_yellow(['BEFORE THRESHOLDING : '])
P_pd = P_psd + ee*eye(d)
disp(['    eigenvalues of P = ', num2str(eig(P_pd)')])
disp(['    eigenvalues of A^TP+PA before thersholding = ', num2str(eig(A'*P_pd+P_pd*A)')])
disp('')
disp_yellow(['AFTER THRESHOLDING (epsilon = 1e-3) : '])
P_pd(abs(P_pd)<1e-3) = 0.0
disp(['    eigenvalues of P = ', num2str(eig(P_pd)')])
disp(['    eigenvalues of A^TP+PA = ', num2str(eig(A'*P_pd+P_pd*A)')])
disp(' ')
disp('press any key to continue ...');
disp(' ');
pause();

% 3(a) : Stabilizable:
disp_title(['3(a) : Stabilizable:'])
cvx_begin sdp quiet
	variable Q_psd(d,d)
	variable s
	s >= eps
	Q_psd - ee*eye(d) == semidefinite(d);
	-A'*Q_psd - Q_psd*A - s*B*B' - ee*eye(d) == semidefinite(d)
cvx_end
Q_pd = Q_psd + ee*eye(d);
K = -s/2*B'*inv(Q_psd)
disp_yellow(['checking if closed-loop is stable <=> A_cl is Hurwitz:'])
A_cl = A + B*K;
disp(['    eigenvalues of A_cl = ', num2str(eig(A_cl)')])
if all(real(eig(A_cl)) < 0)
	disp_red(' >> closed-loop system is stable')
else
	disp_red(' >> closed-loop system is NOT stable')
end
disp(' ')
disp('press any key to continue ...');
disp(' ')
pause();

%% 3(v) : Stabilizable:
disp_title(['3(b) : Stabilizable (change of variables method) :'])
disp(['    >> LMI : AQ + QA'' + BY + Y''B'' < 0 s.t. Q>0 | Y==KQ'])
cvx_begin sdp quiet
	variable Q_psd(d,d)
	variable Y(size(B,2),d)
	Q_psd - ee*eye(d) == semidefinite(d);
	-A*Q_psd - Q_psd*A' - B*Y - Y'*B' - ee*eye(d) == semidefinite(d)
cvx_end
Q_pd = Q_psd + ee*eye(d);
K = Y*inv(Q_pd)

disp_yellow([' >> checking if closed-loop is stable <=> A_cl is Hurwitz:'])
A_cl = A + B*K;
disp(['    eigenvalues of A_cl = ', num2str(eig(A_cl)')])
if all(real(eig(A_cl)) < 0)
	disp_red(' >> closed-loop system is stable')
else
	disp_red(' >> closed-loop system is NOT stable')
end
disp(' ')
disp('press any key to continue ...');
disp(' ')
pause();


disp_title(['5) H_inf norm of linear system :'])
cvx_begin sdp quiet
	variable P_psd(d,d)
	variable g2
	minimize g2
	-[A'*P_psd+P_psd*A+C'*C    P_psd*B;
	  B'*P_psd               -g2*eye(2)] == semidefinite(d+2);
	 g2 >= 1e-6;
	 P_psd - ee*eye(d) == semidefinite(d);
cvx_end
disp_blue(['    Gamma = Bound on H_inf using SDP = ', num2str(sqrt(g2))])


%% compute the infinity norm using matlab:
G = ss(A,B,C,D);
disp_blue(['    Matlab''s norm([A,B,C,D], inf)    = ', num2str(norm(G,inf))]);
disp(' ')
disp('press any key to continue ...');
disp(' ')
pause();

%% Do H2 Optimal controller design:
disp_title(['4) H2 optimal controller design : '])
H2OptimalControl;
