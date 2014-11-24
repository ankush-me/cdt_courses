

%% script to forward-simulate (compute trajectories) given a system of ode's:
a = 5;
b = 5;
c = 4;
d = 8;

f_pp = @(t,Y) [b*Y(1)-a*Y(1)*Y(2);
			   c*Y(1)*Y(2)-d*Y(2)];

y0 = [0; 5];
[ts, ys] = ode23(f_pp, [0, 10], y0);
plot(ts, ys(:,1));
hold on;
plot(ts, ys(:,2));
legend('hare', 'lynx');

