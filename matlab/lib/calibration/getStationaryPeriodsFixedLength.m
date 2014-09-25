%BSD 2-Clause (c) 2014: N.Hammerla, D.Jackson (Newcastle), A.Doherty (Oxford)
function S = getStationaryPeriodsFixedLength(D)
%
%   S = getStationaryPeriodsFixedLength(D)
%
%   Estimate times in which the sensor is not moving. This function
%   estimates the standard deviation per accelerometer axis in epochs. It
%   returns those samples (mean over epoch of 10s) for times the sensor is not
%   moving, to be used in automated calibraction procedure (estimateCalibration.m).
%
%   Input:
%       D   [struct]        As read from AX3_readFile.m, contains at least fields:
%                           .ACC    [one week at 100Hz x 4]
%                           .TEMP   [Mx2]   time [matlab], temp [C]
%
%   Output:
%       S   [Nx5]       Samples time,XYZ,temp from stationary periods (mean over
%                       epoch).
%
%
%   Nils Hammerla, '14
%   <nils.hammerla@ncl.ac.uk>
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

% threshold for activity
actThresh = 0.013;

% initialize
S = [];
A = [];

% construct new matrix from reshaped data
for i=1:3,
    S = [S ; reshape(D.ACC(:,i+1), 10*100, 86400*7/10)]; 
end

% check for activity in each epoch
A = sum([
    std(S(   1:1000,:), 0, 1) <= actThresh
    std(S(1001:2000,:), 0, 1) <= actThresh
    std(S(2001:3000,:), 0, 1) <= actThresh
    ],1);

% get stationary epochs
ind = find(A>=2);

% interpolate temperature (nearest point to middle of epoch)
T = interp1(D.TEMP(:,1),D.TEMP(:,2), 0.5+5/86400+(ind-1)*10/86400,'nearest',0);

% construct return matrix
S = [
    (ind-1)*10/86400
    mean(S(   1:1000,ind), 1)   % mean of X
    mean(S(1001:2000,ind), 1)   % mean of Y
    mean(S(2001:3000,ind), 1)   % mean of Z
	T
    ];

% transpose and we are done
S = S';


end
