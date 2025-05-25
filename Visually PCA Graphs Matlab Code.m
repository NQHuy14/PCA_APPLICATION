clear; close all; clc; %clear the screen

x = linspace(-10, 10, 100)'; % gen data, linspace: create data from -10 to 10 with 100 random data values
noise = 20 * randn(size(x)); % 20 is noise value * with random normal distribution (gen noised data around 0 with size of x)
y = 3 * x + noise; % calc the y axis              
data = [x y]; % created data matrix with x and y

figure; % create the window to draw
scatter(data(:,1), data(:,2), 15, 'g', 'filled'); % draw the dots, scatter(x, y, size, color, style) - data(:,1) is getting the data with all the rows of columns 1
title('1. Original data'); xlabel('X'); ylabel('Y'); axis equal; grid on;

%% 2. Mean-centering
mean_data = mean(data); %calc the average value of data matrix (all the data are calculated)                          
X = bsxfun(@minus, data, mean_data); % minus each data with its average (to make sure the size is smaller but the direction is the same)           

figure;
scatter(X(:,1), X(:,2), 15, 'g', 'filled'); hold on; % hold on to keep the figure not removed
plot([min(X(:,1))-1 max(X(:,1))+1], [0 0], 'k--'); %plot to draw the line plot(start(x,y), end(x,y), style) - k is black
plot([0 0], [min(X(:,2))-1 max(X(:,2))+1], 'k--'); %min(X(:,2)) is to find the min value of all the rows of column 2 (y), max is the same, +1 -1 is just for clarity drawing
title('Data after centered');
xlabel('X'); ylabel('Y'); axis equal; grid on;

%% 3. Covariance matrix and eigenvectors
C = cov(X); %calc the covariance matrix (matlab already had the cov for auto calculating)
[V, D] = eig(C); %calc the eigenvectors and eigenvalues, also built-in function in matlab like cov (V is of vectors, D is values)
[~, idx] = sort(diag(D), 'descend'); % '~' is for 'just igonre it', idx is index, sort the eigenvalue descending
V = V(:, idx);  % change the postion like the index sorted                      
D = diag(sort(diag(D), 'descend')); %update the eigenvalues' postion just like the index

%% 4. Top eigenvectors
scale = 8; 
pc1 = V(:,1) * scale; %for clarity
pc2 = V(:,2) * scale;

figure;
scatter(X(:,1), X(:,2), 15, 'g', 'filled'); hold on;
quiver(0, 0, pc1(1), pc1(2), 0, 'r', 'LineWidth', 2);
quiver(0, 0, pc2(1), pc2(2), 0, 'b', 'LineWidth', 2);
title('PCA axis: PC1 is red and PC2 is blue');
xlabel('X'); ylabel('Y');
legend('Data', 'PC1', 'PC2'); %create information box
axis equal; grid on;

%% 5. Projecting the data
Z = X * V; %create a new axis
explained_var = diag(D) / sum(diag(D)) * 100; %formula to calc the variance

figure;
scatter(Z(:,1), Z(:,2), 15, 'g', 'filled'); hold on;
quiver(0, 0, max(Z(:,1)), 0, 0, 'r', 'LineWidth', 2);
quiver(0, 0, 0, max(Z(:,2)), 0, 'b', 'LineWidth', 2);
text(max(Z(:,1))*0.6, 0.5, sprintf('PC1: %.2f%%', explained_var(1)), 'Color', 'r', 'FontSize', 12);
text(0.5, max(Z(:,2))*0.6, sprintf('PC2: %.2f%%', explained_var(2)), 'Color', 'b', 'FontSize', 12);
title('Data after using PCA');
xlabel('PC1'); ylabel('PC2'); axis equal; grid on;
