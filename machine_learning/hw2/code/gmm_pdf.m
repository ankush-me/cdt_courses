function p = gmm_pdf(x,y,mu,sigma,Pi)

[d,k] = size(mu);
p=0.0;
for i=1:k
	p = p + Pi(i)*mvnpdf([x y], mu(i,:), sigma{i});
end
p = log10(p);