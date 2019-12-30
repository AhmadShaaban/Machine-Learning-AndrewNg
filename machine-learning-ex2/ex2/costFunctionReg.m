function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


X = transpose(X);
	tt = transpose(theta) * X;
	hypoth = sigmoid(tt);
	hypoth = transpose(hypoth);
	temp = ( (-1*y) .* (log(hypoth)) )  -  ( (1-y).*(log(1-hypoth)) );
	J = 1/m * sum(temp,1);
	toti = theta;
	toti(1) = 0;
	toti = toti .^ 2;
	J = J + ( (lambda/(2*m)) * (sum(toti,1)));

	teka = hypoth - y;

	X = transpose(X);
	totii = theta;
	totii(1) = 0;

	grad = transpose(teka) * X;
	grad = (1/m) .* grad;
	grad = transpose(grad);
	
	grad = grad .+ ( (lambda/m) .* totii );



% =============================================================

end
