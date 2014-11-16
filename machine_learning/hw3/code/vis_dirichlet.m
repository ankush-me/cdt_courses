alpha = 5*[2 2 2];
x1 = linspace(0,1,101);
x2 = linspace(0,1,101);
[X1,X2] = ndgrid(x1,x2);
X3 = 1 - X1 - X2;
bad = (X1+X2 > 1); X1(bad) = NaN; X2(bad) = NaN; X3(bad) = NaN;

betaConst = exp(sum(gammaln(alpha))-gammaln(sum(alpha)));
F = (X1.^(alpha(1)-1) .* X2.^(alpha(2)-1) .* X3.^(alpha(3)-1)) / betaConst;

figure, surf(X1,X2,F,'EdgeColor','none');
xlabel('x1'); ylabel('x2'); zlabel('f(x1,x2,1-x1-x2)');
view(-160,40);
figure, surf(X1,X2,X3,F,'EdgeColor','none');
xlabel('x1'); ylabel('x2'); zlabel('x3');
view(-160,40);
