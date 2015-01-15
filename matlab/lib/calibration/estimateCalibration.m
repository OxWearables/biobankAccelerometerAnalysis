%BSD 2-Clause (c) 2014: N.Hammerla, D.Jackson (Newcastle), A.Doherty(Oxford)
function estimate = estimateCalibration(data, varargin)
%
%   estimate = estimateCalibration(M, varargin)
%
%   Use data samples in M to estimate correct scaling, offset and
%   temperature offset for accelerometers (calibration). Use
%   getStationaryPoints.m to get samples that reflect stationary periods
%   suitable for this estimation. Returns struct with estimates.
%
%   Input:
%       M       [Nx4] or [Nx5]      Matrix with samples that reflect stationary 
%                                   periods. Format: [time, X, Y, Z, (temp.)]
%
%   Optional arguments ('name', value):
%       useTemp     0/1             Use temperature for estimation?
%                                   (default: 1)
%       maxIter     int             Max number of iterations for optimization
%                                   (default: 100)
%       convCrit    double          Criterion for convergence (derivative
%                                   of scale-factors in subsequent iterations)
%                                   (default: 0.001)
%       verbose     0/1             Be verbose about it? (default: 0)
%
%   Output:
%       estimate.scale
%       estimate.offset
%       estimate.tempOffset
%       estimate.error
%       estimate.referenceTemperature
%       estimate.referenceTemperatureQuantiles
%
%   Use rescaleData.m to apply these estimates to your data!
%
%   v0.2
%   Nils Hammerla '14
%   <nils.hammerla@ncl.ac.uk>
%

% Inspired by code from GGIR package (http://cran.r-project.org/web/packages/GGIR/index.html)
%   Vincent T van Hees, Zhou Fang, Jing Hua Zhao

% 
% Copyright (c) 2014, Newcastle University, UK.
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are met: 
% 1. Redistributions of source code must retain the above copyright notice, 
%    this list of conditions and the following disclaimer.
% 2. Redistributions in binary form must reproduce the above copyright notice, 
%    this list of conditions and the following disclaimer in the documentation 
%    and/or other materials provided with the distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
% POSSIBILITY OF SUCH DAMAGE. 
% 

estimate = struct();
estimate.reliable = 1;
p = inputParser;
    
% define optional arguments with default values
addOptional(p,'useTemp',      1, @isnumeric); 
addOptional(p,'maxIter',    100, @isnumeric); 
addOptional(p,'convCrit', 0.001, @isnumeric); 
addOptional(p,'verbose',      0, @isnumeric); 

% parse inputs
parse(p,varargin{:});
p = p.Results;

if isempty(data), warning('empty input matrix M! Aborting'); return; end

% ditch time
data = data(:,2:end);

% remove any nans
n = isnan(data);
data = data(sum(n,2)==0,:);

N = size(data,1);

% abort if < 4 points
if N < 4, warning('Not enough points for estimation, aborting'); estimate.reliable = 0; return; end
% warn if less than 10 points
if N < 10, warning('Just 10 points or less, estimation may not be accurate'); estimate.reliable = 0; end

if p.useTemp
    % data in first columns, end is temperature
    D = data(:,1:end-1);
    temp = data(:,end);
    meanTemp = mean(temp);
else
    % no temperature used, just use "dummy" parameters (zero temp)
    if size(data,2)==3,
        D = data;
    else
        D = data(:,1:3);
    end
    temp = zeros(N,1);
    meanTemp = 0;
end

%temp = temp - meanTemp;

% initialise variables
scale = ones(1,size(D,2));
offset = zeros(1,size(D,2));
tempOffset = offset;
weights = ones(size(D,1),1);

% zero mean temperature
temp = temp - meanTemp;

% save input data
D_in = D;

% check how well the sphere is populated 
sid = 0;
for i=1:3
    if min(D(:,i)) <= -0.3 && max(D(:,i)) >= 0.3,
        sid = sid + 1;
    end
end

if     sid < 1,
    warning('Unit sphere is not well populated with samples! No axis fits criterion. Do not use estimate for correction.');
    estimate.reliable = 0;
elseif sid == 1,
    warning('Unit sphere is not well populated with samples! Just one axis fits criterion. Do not use estimate for correction.');
    estimate.reliable = 0;
elseif sid == 2,
    warning('Unit sphere is not well populated with samples! Two axis fulfill criterion. Beware estimate may not be of good quality.');
    estimate.reliable = 0;
end

% main loop to estimate unit sphere
for i=1:p.maxIter,
    % scale input data with current parameters
    % model: (offset + D_in) * scale + D_in * T * tempOffset)
    %D  = (repmat(offset,N,1) + D_in) .* repmat(scale,N,1) + D_in .* repmat(temp,1,3) .* repmat(tempOffset,N,1);
    
    % model: (offset + D_in) * scale + T * tempOffset)
    D  = (repmat(offset,N,1) + D_in) .* repmat(scale,N,1) + repmat(temp,1,3) .* repmat(tempOffset,N,1);
    
    % targets: points on unit sphere
    target = D ./ repmat(sqrt(sum(D.^2,2)),1,size(D,2));
    
    % initialise vars for optimisation
    gradient = zeros(1,size(D,2));
    off = gradient;
    tOff = gradient;
    
    % do linear regression per input axis to estimate scale offset
    % (and tempOffset)
    for j=1:size(D,2),
        if p.useTemp,
            mdl = LinearModel.fit([D(:,j) temp], target(:,j), 'linear', 'Weights', weights);
        else
            mdl = LinearModel.fit([D(:,j)], target(:,j), 'linear', 'Weights', weights);
        end
        coef = mdl.Coefficients.Estimate;
        off(j) = coef(1);       % offset     = intersect
        gradient(j) = coef(2);  % scale      = gradient
        
        if p.useTemp,
            tOff(j) = coef(3);  % tempOffset = last coeff
        end
    end
    
    % change current parameters
    sc = scale; % save this for convergence comparison
    offset = offset + off ./ (scale .* gradient);% ./ scale; % adapt offset
    scale = scale .* gradient;  % adapt scaling
    
    if p.useTemp
        % apply temperature offset 
        tempOffset = tempOffset .* gradient + tOff; 
    end
    
    % weightings for linear regression
    % read: ignores outliers
    % overall limited to a maximum of 100
    weights = min([1 ./ sqrt(sum((D-target).^2,2)), repmat(100,N,1)],[],2);
   
    % no more scaling change -> assume it has converged
    cE = sum(abs(scale-sc));
    converged = cE < p.convCrit;
    
    % L2 error to unit sphere
    E = sqrt(mean(sum((D-target).^2),2));
    
    if converged,
        break % get out of this loop
    end
    
    if i==p.maxIter && p.verbose == 1,
        % no convergence but assume that we are done anyway
        fprintf('Maximum number of iterations reached without convergence.\n');
    end
    
    if p.verbose == 1,
        fprintf('iteration %d\terror: %.4f\tconvergence: %.6f\n',i,E,cE);
    end
end

if p.verbose
    fprintf('Estimated parameters:\n\n');

    fprintf('\tScaling:\t%.3f %.3f %.3f\n', scale);
    fprintf('\tOffset:\t\t%.3f %.3f %.3f\n', offset);
    if p.useTemp,
        fprintf('\tTemp-Offset:\t%.3f %.3f %.3f\n', tempOffset);
    end
end

% assign output
estimate.scale = scale;
estimate.offset = offset;
estimate.tempOffset = tempOffset;
estimate.error = E;
if p.useTemp == 1,
    estimate.referenceTemperature = meanTemp;
    estimate.referenceTemperatureQuantiles = quantile(temp,[0.05 0.95]);
else
    estimate.referenceTemperature = 0;
    estimate.referenceTemperatureQuantiles = quantile(temp,[0 0]);
end
end
