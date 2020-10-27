% Example using kNN Regressor

% Generate sample data
X = sort(5 * rand(40,1));
Xnew = sort(5 * rand(500,1));
Y = sin(X)';

% Add noise to targets
Y = Y + (0.5 - rand(1,40));

k = 5;
metric = 'euclidean';
weights = {'uniform', 'distance'};
for i = 1:2
    mdl = kNNeighborsRegressor(k,metric,weights(i));
    mdl = mdl.fit(X,Y);
    Ypred = mdl.predict(Xnew);
    subplot(2,1,i)
    plot(X,Y,'o',Xnew,Ypred)
    legend('data','prediction')
    title(strcat('kNNeighborsRegressor (k = 5, metric = ''euclidean'', weights = ''', weights(i), ''')'))
end

saveas(gca, 'Regression_results', 'png')
