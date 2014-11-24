%BSD 2-Clause (c) 2014: N.Hammerla, D.Jackson (Newcastle), A.Doherty (Oxford)
function D = readInterpolateCalibrate(filePath, outputPath)
%
%   D = readInterpolateCalibrate(FILEPATH, OUTPUTPATH)
%
%   Reads cwa file in FILEPATH. Performs interpolation of accelerometer
%   signals to 100Hz, finds stationary points and peforms a calibration.
%   Writes output to .WAV (or .FLAC) file in OUTPUTPATH. Format of that
%   output file is:
%       Track 1: X values / 8g
%       Track 2: Y values / 8g
%       Track 3: Z values / 8g
%       Track 4: Empty, except for sporadic Temperature and Light readings:
%           If entry is not zero, positive value indicates temperature and
%           negative value indicates a light-sensor reading. Position in
%           File reflects time of measurement. Scaling:
%           - Temperature / 100
%           - Light / 1000
%
%   Nils Hammerla
%   <nils.hammerla@ncl.ac.uk>
%

addpath(genpath('lib'));

% read file info (start-time / end-time, etc)
fprintf('reading file\n');
info = AX3_readFile(filePath, 'useC', 1, 'info', 1, 'ignoreSessionId', 1);
% read in one week of data from start
D = AX3_readFile(filePath, 'useC', 1, 'validPackets', info.validPackets, ...
    'startTime', floor(info.start.mtime), 'stopTime', ceil(info.start.mtime)+7, ...
    'ignoreSessionId', 1); % SessionIds require fixing! 

% pre-processing
T = linspace(10/24,7+10/24,86400*100*7);  % exactly one week at 100Hz from 10:00 to 10:00
st = D.ACC(1,1);                            % start-time
D.ACC(:,1)   = D.ACC(:,1)   - floor(st);    % relative timestamps to start of first day
D.TEMP(:,1)  = D.TEMP(:,1)  - floor(st);    % relative timestamps
D.LIGHT(:,1) = D.LIGHT(:,1) - floor(st);    % relative timestamps

% fix problematic timestamps in D (same time-stamp for subsequent samples)
D.ACC = D.ACC([diff(D.ACC(:,1)) > 0 ; 1] > 0,:);
D.TEMP = D.TEMP([diff(D.TEMP(:,1)) > 0 ; 1] > 0,:);
D.LIGHT = D.LIGHT([diff(D.LIGHT(:,1)) > 0 ; 1] > 0,:);

% interpolate
AC = zeros(length(T),4); % wastes a column for time at the moment
AC(:,1)=T;
for i=2:4,
    AC(:,i) = interp1(D.ACC(:,1),D.ACC(:,i),T,'linear',nan);
end
D.ACC = AC; clear AC;

% run calibration
fprintf('calibrating\n');
% get stationary points
S = getStationaryPeriodsFixedLength(D); % 10s epochs, 0.013g std threshold
% estimate scaling, offset, temp-offset
D.calibration = estimateCalibration(S, 'useTemp', 1); 

% just calibrate if the estimate is good
% -> Sufficient number of stationary periods 
% -> Stationary periods well distributed around unit sphere
if D.calibration.reliable == 1,
    % get clipping samples (value == 8)
    indp = find(D.ACC(:) ==  8);
    indn = find(D.ACC(:) == -8);
    
    % apply calibration
    % May throw warnings if estimate is supposedly unreliable
    D = rescaleData(D, D.calibration);
    
    % clip back to 8
    D.ACC(indp) =  8;
    D.ACC(indn) = -8;
    % some values may still be clipping after calibration (that did not
    % clip before). Leave them in for now.
end

% write output as wav file
fprintf('writing output\n');
writeOutput(D, info, outputPath);

end

function writeOutput(D, info, outputPath)

% change possible nans to 
D.ACC(sum(isnan(D.ACC(:,2:4)),2) > 0,2:4)=0;

% scale from -1 to 1
D.ACC(:,2:4) = D.ACC(:,2:4) ./ 8; 

% transform temperature
T = zeros(length(D.ACC),1);
% from matlab time to sample position
D.TEMP(:,1) = round((D.TEMP(:,1)-0.5) * 86400 * 100); 
% find invalid readings (outside of our range)
ind = find(D.TEMP(:,1) > 0);
D.TEMP = D.TEMP(ind,:); D.LIGHT = D.LIGHT(ind,:);
% scale
T(D.TEMP(:,1)) = D.TEMP(:,2) ./ 100;
T(D.TEMP(:,1)+1) = -D.LIGHT(1:length(D.TEMP),2) ./ 1000;
% prevent overflow
T = T(1:length(D.ACC)); 

% write file
audiowrite(outputPath, [D.ACC(:,2:4) T], 16000, ...
    'Title', num2str(info.deviceId), 'Artist', datestr(info.start.mtime,'yyyy-mm-dd HH:MM:SS.FFF'), 'Comment', '', ...
    'BitsPerSample', 16);

end
