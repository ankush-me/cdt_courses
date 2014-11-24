
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

ev = eig(A);
scatter(real(ev), imag(ev), 1000, '.');
xlabel('real(\lambda)');
ylabel('imag(\lambda)');
grid on;


%%% if the roots are : (a+ib) and (a-ib) => damping factor = -a/sqrt(a^2 + b^2):
f_damp = -real(ev)./abs(ev);
disp(['damping factors : ', num2str(f_damp(1)), ',  ',  num2str(f_damp(3))]);


