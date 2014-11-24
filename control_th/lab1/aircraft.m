
%% Case Study : Control of Spacecraft & Aircraft

%%% PART A:
%===========

A = [-0.003    0.039     0.000   -0.322;
	 -0.065   -0.319     7.740    0.000;
	  0.020   -0.101    -0.429    0.000;
	  0.000    0.000     1.000    0.000];

B = [ 0.01    1.000;
     -0.18   -0.040;
     -1.16    0.598;
      0.00    0.000 ];

C = eye(4);
D = zeros(4,2);

ev = eig(A);
scatter(real(ev), imag(ev), 1000, '.');
xlabel('real(\lambda)');
ylabel('imag(\lambda)');
grid on;

%% simulate the system:
%----------------------
x0 = [0; 0; 0; 0.1];
t  = 0:1:6000;
u  = zeros(size(t,2), 2);
sys = ss(A,B,C,D);
[~,~,X] = lsim(sys, u, t, x0);

figure(2);

subplot(2,1,1); hold on;
m1 = real(exp(ev(2)*t));
m2 = real(exp(ev(4)*t));
plot(t, m1);
plot(t, m2);
xlabel('t'); ylabel('exp(\lambda t)');
title(['Note how the first eigenvalue (\lambda_1) dies out quickly => Dominant eigenvalue = \lambda_2 = ' num2str(ev(4))]);
legend('\lambda_1', '\lambda_2');

subplot(2,1,2);
plot(t, X);
xlabel('t'); title('State-space variables with zero inputs and x0 = [0,0,0,0.1]');
legend('\Delta u', '\Delta w', '\Delta q', '\Delta\theta');

%%% if the roots are : (a+ib) and (a-ib) => damping factor = -a/sqrt(a^2 + b^2):
f_damp = -real(ev)./abs(ev);
disp(['Damping factors : ', num2str(f_damp(1)), ',  ',  num2str(f_damp(3))]);
disp(['Dominant mode corresponds to the eigenvalue with smaller damping ratio.'])
disp(['  => This eigen-value is ' num2str(ev(4))])


