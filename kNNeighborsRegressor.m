classdef kNNeighborsRegressor
%k-Nearest Neighbors Regressor
%
% Author: M.Sc. David Ferreira - Federal University of Amazonas 
% Contact: ferreirad08@gmail.com
% Date: September 2020

properties
    k = 5 % Number of neighbors
    metric = 'euclidean' % Distance metric
    weights = 'uniform'  % Weight function
    X
    Y
end
methods
    function obj = kNNeighborsRegressor(k,metric,weights)
        if nargin > 0
            obj.k = k;
        end
        if nargin > 1
            obj.metric = metric;
        end
        if nargin > 2
            obj.weights = weights;
        end
    end
    function obj = fit(obj,X,Y)
        obj.X = X;
        obj.Y = Y(:); % Column vector
        if obj.k < 2, obj.Y = Y(:)'; end % Row vector
    end
    function Ypred = predict(obj,Xnew)
        distances = pdist2(obj.X,Xnew,obj.metric); % Euclidean distances matrix
        [distances,indices] = sort(distances); % Ordered distances
        Ynearest = obj.Y(indices(1:obj.k,:)); % k-nearest labels
        if strcmp(obj.weights,'uniform')
            w = ones(size(Ynearest)); % For simple mean
        elseif strcmp(obj.weights,'distance')
            w = 1./distances(1:obj.k,:); % For weighted mean
        end
        Ypred = sum(w.*Ynearest,1)./sum(w,1);
    end
end
end
