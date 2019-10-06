function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
%diffX2 = max(max(X(:,2))) - min(min(X(:,2)));
%meanX2 = mean(X(:,2));
%tempX2 = X(:,2) - meanX2;
%temptempX2 = tempX2 ./diffX2;
%neX(:,2) = temptempX2;
%neX(:,1) = X(:,1);
%H = [0; 0];
%theta=[0 ; 0];
%delete_t = [0 ; 0];
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
%delta = zeros(1, 2);
%delta = 0.5/m * (theta' * X' - y) * X;
%theta = theta - alpha * delta;
H = X * theta;
delete_t = theta  - (alpha/m) * (X' * (H - y));
theta = delete_t;

    % ============================================================

    % Save the cost J in every iteration
    
    
    J_history(iter) = computeCost(X, y, theta);

end

end
