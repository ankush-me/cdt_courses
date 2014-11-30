close all;
clear;

syms x1 x2;
x = [x1;x2]

%% ellipsoid 1:
A1 = 2*randn(2,2);
A1 = A1'*A1;
%A1 = [ 0.1355    0.1148;  0.1148    0.4398];
xc1= [0;0]; %% center of the ellipse
b1 = -(xc1'*A1)';
c1 = xc1'*A1*xc1-1;
E1 = [A1 b1;
      b1' c1];
e1 = x.'*A1*x + 2*b1'*x + c1;

%% ellipsoid 2:
A2 = 2*randn(2,2)/4;
A2 = A2'*A2;
%A2 = [ 0.6064   -0.1022; -0.1022    0.7344];
xc2= [1;2]; %% center of the ellipse
b2 = -(xc2'*A2)';
c2 = xc2'*A2*xc2-1;
E2 = [A2 b2;
      b2' c2];
e2 = x.'*A2*x + 2*b2'*x + c2;

E = {E1, E2};
n = size(E,2);
d = 2;

cvx_begin sdp

	variables g t xc(2) tau(n) Ec(d+1,d+1)

	minimize t
	
	%% ball subsumes others:
	Ec == [eye(d)  -xc; -xc'  g];
	tau(1)*E1 - Ec == semidefinite(d+1);
	tau(2)*E2 - Ec == semidefinite(d+1);

	%% constraint on the radius:
	[eye(d)  xc;
	 xc'     t+g] == semidefinite(d+1);

	tau(1) >= 0;
	tau(2) >= 0;

cvx_end

g
xc
r = sqrt(xc'*xc - g)
tau

Ec = full(Ec);
Ap = Ec(1:2,1:2);
bp = Ec(1:2,3);
cp = Ec(3,3);
ec = x.'*Ap*x + 2*bp'*x + cp;


% noangles = 200;
% angles   = linspace( 0, 2 * pi, noangles );
% Ai = A2
% bi = A2\b2
% alpha = b2'*inv(A2)*b2 - c2
% ellipse  = Ai \ [ sqrt(alpha)*cos(angles)-bi(1) ; sqrt(alpha)*sin(angles)-bi(2) ]
% e
% plot( ellipse(1,:), ellipse(2,:), 'b-' );


hold on;
axis equal;
ezplot(e1==0);
ezplot(e2==0);
p = ezplot([x.' 1]*Ec*[x;1]==0);
p.Color='red';
