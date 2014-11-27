close all;
clear;

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

%% Q1 : solve for P : P>0, A'P + PA < 0
ee = 1e-4;
d  = size(A,1);
cvx_begin sdp quiet
	variable P_psd(d,d)
	-A'*P_psd - P_psd*A -ee*eye(d) == semidefinite(d)
	P_psd - ee*eye(d) == semidefinite(d);
cvx_end
P_pd = P_psd + ee*eye(d)
disp(['eigenvalues of P = ', num2str(eig(P_pd)')])
disp(['eigenvalues of A^TP+PA = ', num2str(eig(A'*P_pd+P_pd*A)')])
disp('press any key to continue ...');
disp(' ')
pause();


% %% 2(a) : Diagonal Lyapunov:
% disp(['Diagonal lyapunov:'])
% d  = size(A,1);
% cvx_begin sdp quiet
% 	variable p(d)
% 	-A'*(diag(p)-ee*eye(d)) - (diag(p)-ee*eye(d))*A == semidefinite(d)
% 	p >= 0
% cvx_end
% P_pd = diag(p) + ee*eye(d)
% disp(['eigenvalues of P = ', num2str(eig(P_pd)')])
% disp(['eigenvalues of A^TP+PA = ', num2str(eig(A'*P_pd+P_pd*A)')])
% disp('press any key to continue ...');
% pause();


%% 2(a) : Diagonal Lyapunov:
disp(['Sparse-Heuristic Diagonal lyapunov:'])
d  = size(A,1);
cvx_begin sdp quiet
	variable P_psd(d,d)
	minimize(norm(P_psd(:), 1))
	-A'*P_psd - P_psd*A -ee*eye(d) == semidefinite(d)
	P_psd - ee*eye(d) == semidefinite(d);
cvx_end
P_pd = P_psd + ee*eye(d)
disp(['BEFORE THRESHOLDING : '])
disp(['    eigenvalues of P = ', num2str(eig(P_pd)')])
disp(['    eigenvalues of A^TP+PA before thersholding = ', num2str(eig(A'*P_pd+P_pd*A)')])
P_pd(abs(P_pd)<1e-3) = 0.0;
disp(['AFTER THRESHOLDING (epsilon = 1e-3) : '])
disp(['    eigenvalues of P = ', num2str(eig(P_pd)')])
disp(['    eigenvalues of A^TP+PA = ', num2str(eig(A'*P_pd+P_pd*A)')])
disp('press any key to continue ...');
disp(' ');
pause();
