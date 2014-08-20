function D = rescaleData(D, e)
%
%   D = rescaleData(D, calibrationEstimate)
%
%   Rescale the accelerometer data in D (struct returned from
%   AX3_readFile.m) using the calibration estimates, calculated using
%   estimateCalibration.m
%
%   Input:
%       D                   struct      Struct with data as read by AX3_readfile, contains
%                                       at least .ACC = [Nx4] double (TXYZ) and .TEMP =
%                                       [Mx2] double (T,temperature).
%
%       calibrationEstimate struct      Struct with fields
%                                       .scale      [1x3] double scale-factor
%                                       .offset     [1x3] double offset
%                                       .tempOffset [1x3] double temp. offset
%                                       .referenceTemperature [1x1] double reference temperature
%                                       .referenceTemperatureQuantiles [1x2] 10th / 90th percentile (temp-meanTemp)
%
%   The acceleromter data is rescaled as follows:
%
%   sig = (sig + offset) * scale + (temp-refTemp) * tempOffset
%
%
%   Nils Hammerla '14
%   <nils.hammerla@ncl.ac.uk>
%
% v0.2 
%

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

N = size(D.ACC(:,1));

if isfield(D,'TEMP'), % are there temperature readings?
    % get rid of bad readings and timestamps
    D.TEMP = D.TEMP(D.TEMP(:,2)>0,:);
    D.TEMP = D.TEMP(find(diff(D.TEMP(:,1))>0),:);
    
    % interpolate temperature
    T = interp1(D.TEMP(:,1),D.TEMP(:,2),D.ACC(:,1),'pchip',e.referenceTemperature);
else
    % no temperature!
    if e.referenceTemperature > 0,
        warning('No temperature given. Assuming reference temperature. Results may be inaccurate.');
    end
    % add pseudo-time
    T = repmat(e.referenceTemperature,[N,1]);
end

% sanity check: mean temperature within calibrated range?
if sum(e.referenceTemperatureQuantiles ~= 0) > 0,
    if mean(T) < e.referenceTemperature + e.referenceTemperatureQuantiles(1) | ...
       mean(T) > e.referenceTemperature + e.referenceTemperatureQuantiles(2),
        warning('Mean temperature outside of calibrated temperature range (10th and 90th percentiles). Results may be inaccurate.');
    end
end

% substract reference time
T = T - e.referenceTemperature; % latter may be zero, but then e.tempOffset is zero as well

% apply estimates
D.ACC(:,2:4) = (repmat(e.offset,[N,1]) + D.ACC(:,2:4)) .* repmat(e.scale,[N,1]) ...
                 + repmat(T,[1,3]) .* repmat(e.tempOffset,[N,1]);

end