
syms th p dth dp m M l g;

M = [ (M+m),          -m*l*cos(th);
	  -m*l*cos(th),    m*l^2       ];
K = [0;   -m*l*sin(th)*(dp*dth + g)];

G = -inv(M)*K

dG_dp   = diff(G, p);
dG_ddp  = diff(G, dp);
dG_dth  = diff(G, th);
dG_ddth = diff(G, dth);

J_G = [dG_dp dG_ddp dG_dth dG_ddth]

G = subs(G, [p, dp, th, dth], [0,0,0,0])
J = subs(J_G, [p, dp, th, dth], [0, 0, 0, 0])
B = subs(inv(M), [p, dp, th, dth], [0,0,0,0])*[1;0];

G_full = [zeros(2,1); G]
J_full = [zeros(2), eye(2); J]
B_full = [zeros(2,1); B]

%disp(['dx/dt ='  num2str(G_full) '+' num2str(G_full) 'x + ' num2str(B_full) 'u'])
