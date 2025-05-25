clearvars; close all; clc;

T = readtable('train.csv');

y = T.critical_temp;
T.critical_temp = [];
X = table2array(T);

y = log10(y);

[Xs, mu, sigma] = zscore(X);

[coeff, score, latent, ~, explained] = pca(Xs, 'Algorithm','svd');
K = find(cumsum(explained) >= 95, 1);
Z = score(:, 1:K);
fprintf('Retaining %d / 81 components (≥95%% variance)\n', K);

X_baseline = [ones(size(Xs,1),1) Xs];
beta_full  = X_baseline \ y;
y_full     = X_baseline * beta_full;
rmse_full  = sqrt(mean((y - y_full).^2));

X_pca   = [ones(size(Z,1),1) Z];
beta_pca = X_pca \ y;
y_pca    = X_pca * beta_pca;
rmse_pca = sqrt(mean((y - y_pca).^2));

SST = sum((y - mean(y)).^2);
SSE_full = sum((y - y_full).^2);
SSE_pca = sum((y - y_pca).^2);
r2_full = 1 - SSE_full/SST;
r2_pca = 1 - SSE_pca/SST;

fprintf('\n============= ANALYSIS RESULTS =============\n');
fprintf('Original data dimensions: %d\n', size(Xs, 2));
fprintf('Principal components retained (K): %d\n', K);
fprintf('Dimensionality reduction ratio: %.1f%%\n', (1-K/size(Xs,2))*100);
fprintf('Variance retained: ≥ 95%%\n\n');

fprintf('Model performance comparison:\n');
fprintf('%-40s: %.3f\n', 'RMSE with all features', rmse_full);
fprintf('%-40s: %.3f\n', sprintf('RMSE with PCA (%d components)', K), rmse_pca);
fprintf('%-40s: %.3f\n', 'R² with all features', r2_full);
fprintf('%-40s: %.3f\n', sprintf('R² with PCA (%d components)', K), r2_pca);