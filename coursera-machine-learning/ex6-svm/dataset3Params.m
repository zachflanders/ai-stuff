function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
params_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
results = zeros(length(params_vec) ^ 2, 3);
attempt = 0;
for i = 1:length(params_vec)
    for j = 1:length(params_vec)
        attempt = attempt + 1;
        fprintf('Finding optimal C and sigma: training model %f of %f\n', attempt, length(params_vec)^2);
        model = svmTrain(X, y, params_vec(i), @(x1, x2) gaussianKernel(x1, x2, params_vec(j)));
        predictions = svmPredict(model, Xval);
        pred_error = mean(double(predictions ~= yval));
        results(attempt, 1) = params_vec(i);
        results(attempt, 2) = params_vec(j);
        results(attempt, 3) = pred_error;
    end
end
[min_pred, min_pred_idx] = min(results(:, 3));
C = results(min_pred_idx, 1);
sigma = results(min_pred_idx, 2);
fprintf('Optimal C and sigma: %f , %f\n', C, sigma);






% =========================================================================

end
