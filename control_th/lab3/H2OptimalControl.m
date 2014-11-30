function H2OptimalControl()
A = [-0.003    0.039     0.000   -0.322;
	 -0.065   -0.319     7.740    0.000;
	  0.020   -0.101    -0.429    0.000;
	  0.000    0.000     1.000    0.000];

B2 = [ 0.01    1.000;
      -0.18   -0.040;
      -1.16    0.598;
       0.00    0.000 ];

C1 = [0 1 0 0;
      0 0 1 0];

nx = size(A,1);
nu = size(B2,2);
nz = size(C1,1);
ny = nx;
nw = 1;

B1 = randn(nx,nw);
C2 = eye(ny);

D11 = zeros(nz,nw);
D12 = zeros(nz,nu);
D21 = zeros(ny,nw);
D22 = zeros(ny,nu);

B = [B1 B2];

ee = 1e-4;
cvx_begin sdp quiet
	variables X(nx,nx) Z(nu, nx) W(nu, nu) S(nz, nx);

	minimize trace(W)

	%% X > 0
	X - ee*eye(nx) == semidefinite(nx);

	-A*X - B2*Z - X*A' - Z'*B2' - B1*B1' - ee*eye(nx) == semidefinite(nx);

	S == (C1*X + D12*Z);
	[  X  S';
	   S  W ] - ee*eye(nx+nu) == semidefinite(nx+nu);

cvx_end

disp(['    Since B1 is randomized, the problem might become infeasible.'])
disp(['    Try re-running the script if an infeasible instance is found.'])
disp(' ')
disp(['     SDP status : ' cvx_status])
K = Z*inv(X)
disp(['      ** Tr(W) = ' num2str(trace(W))])
disp(['      ** Tr((C1X+D12Z)inv(X)(C1X+D12Z)'') = ' num2str(trace(S*inv(X)*S')) ' < Tr(W)'])
end


