%% Simulate a spring-mass system:
%% ------------------------------
m = 1;
c = 1.5;
k = 2;

%% input parameters:
%% -----------------
f = 1/40; %% frequency
a = 5; %% amplitude
t = 0:0.001:100;
u = a*(square(2*pi*f*t));
plot(t,u);
pause();

%% Define the system:
%% ------------------
A = [0 1; -k/m -c/m];
B = [0; 1];
C = [0 0];
D = 0;
sys = ss(A,B,C,D);
[~,~,X] = lsim(sys, u, t);

plot(X(:,1), X(:,2));

figure; plot(t, X(:,1));
figure; plot(t, X(:,2));



