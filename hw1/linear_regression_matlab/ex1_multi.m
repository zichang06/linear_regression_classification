%%  Exercise 1: Linear regression with multiple variables
%% ================ Part 1: Feature Normalization ================
% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

% Load Data
data = csvread('train.csv', 1, 2);
featureNum = 384; % 383+1
X = data(:, 1:383);
y = data(:, 384);
m = length(y);

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

% [X mu sigma] = featureNormalize(X); 本题数据已经归一化，所以无需再归一化，如果归一化甚至出现NaN列

% Add intercept term to X 已经帮我们完成了这一步！！
X = [ones(m, 1) X]; 


%% ================ Part 2: Gradient Descent ================
fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.0001;
num_iters = 2000;

% Init Theta and Run Gradient Descent 
theta = zeros(featureNum, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');
%% ================ Part 3: Normal Equations ================
%heta = normalEqn(X, y);

%% ================ Part 4: Predict ================
% Load Data
testData = csvread('test.csv', 1, 2);
X_test = data(:, 1:383);
featureNum = 384; % 383+1
[m,~] = size(testData);
X_test = [ones(m, 1) X]; 
y_predict = X*theta;
id = (0:m-1)';
%% 
% write data into csv
column = {'id', 'reference'};
result = table(id, y_predict, 'VariableNames', column);
writetable(result, 'y_predict.csv');