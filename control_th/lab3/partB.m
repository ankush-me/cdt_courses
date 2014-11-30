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

d = size(A,1);
%% Controllability:
disp(['Controllability : ']);
disp(['   (a) [delta_e] ']);
B1 = B(:,1);
AB_ctr = ctrb(A,B1);
rk = rank(AB_ctr);
if  rk==d
	disp('      --> is controllable');
else
	disp('      --> is NOT controllable');
end
disp(' ');

disp(['   (b) [delta_T] ']);
AB_ctr = ctrb(A,B(:,2));
rk = rank(AB_ctr);
if  rk==d
	disp('      --> is controllable');
else
	disp('      --> is NOT controllable');
end
disp(' ');

disp(['   (b) [delta_e delta_T]''']);
AB_ctr = ctrb(A,B);
rk = rank(AB_ctr);
if  rk==d
	disp('      --> is controllable');
else
	disp('      --> is NOT controllable');
end
disp(' ');
disp('press any key to conitnue...');
disp(' ');
pause();


%% Observability:
disp(['Observability : '])
disp(['   (a) [delta_theta] '])
C = [0 0 0 1];
AC_obs = obsv(A,C);
disp(['         C = ', num2str(C)]);
rk = rank(AC_obs);
if  rk==d
	disp('      --> is observable')
else
	disp('      --> is NOT observable')
end
disp(' ')

disp(['   (b) [delta_w delta_theta]'''])
C = [0 1 0 0;
     0 0 0 1];
disp(['         C = ', num2str(C(1,:))]);
disp(['             ', num2str(C(2,:))]);
AC_obs = obsv(A,C);
rk = rank(AC_obs);
if  rk==d
	disp('      --> is observable')
else
	disp('      --> is NOT observable')
end
disp(' ')
disp('press any key to conitnue...')
disp(' ')
pause();

%% Lyapunov Stability:
disp('Lyapunov Stability:')
Q = eye(d);
P = lyap(A,Q);
ev = eig(P);
disp(['   Eigen-values of P : ', num2str(ev')]);

if all(ev>0)
	disp('    --> P is PD => the system is Asymptotically (in fact exponentially) stable.')
	disp('          Verifying...');
	disp(['             Eigen-values of A are: ', num2str(eig(A)')])	
	disp(['             Eigen-values of A have negative real-part.'])
else
	disp('    --> P is not PD => stability cannot be established.')
end

