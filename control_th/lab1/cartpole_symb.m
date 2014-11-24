
syms t m M l g;
p = sym('p(t)');
th = sym('th(t)');
dp = diff(p,t);
dth = diff(th, t);

M = [ (M+m),          -m*l*cos(th);
	  -m*l*cos(th),    m*l^2       ];
K = [0;   -m*l*sin(th)*(dp*dth + g)];

G = -inv(M)*K;

dG_dp   = diff(G, p);
dG_ddp  = diff(G, dp);
dG_dth  = diff(G, th);
dG_ddth = diff(G, dth);




