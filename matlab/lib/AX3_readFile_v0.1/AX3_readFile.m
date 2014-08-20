function data = AX3_readFile(filename, varargin)
    %
    %   DATA = AX3_readFile(FILENAME, [OPTIONS])
    %
    %   Reads in binary file as produced by AX3 accelerometer. Returns a
    %   struct filled with data from different modalities (ACC, LIGHT,
    %   TEMP). Relies on two external mex-functions (parseValueBlock.c,
    %   parseDate.c) that have to be compiled using the mex interface.
    %
    %       Input arguments:
    %           FILENAME            Path to AX3 sensor file
    %
    %       OPTIONS:
    %
    %           'info'              If set to 1, just reads information
    %                               from file. Such as start time of
    %                               logging, sensor id, etc. (see example
    %                               below)
    %
    %           'validPackets'      A Nx3 matrix containing pre-read valid
    %                               packet locations, timestamps and
    %                               timestamp offsets (produced by
    %                               info-mode). Providing this will heavily
    %                               improve runtime! 
    %
    %           'startTime'         Use this combined with 'stopTime' for
    %                               sliced reading. I.e. read all samples
    %                               from startTime to stopTime. Has to be
    %                               given in Matlab-time format (see
    %                               example below)
    %
    %           'stopTime'          See above.
    %
    %           'modality'          A three element vector [1, 1, 1] that
    %                               indicates which sensor modalities to
    %                               extract. Order is ACC, LIGHT, TEMP. To
    %                               extract ACC and TEMP use [1, 0, 1].
    %
    %           'verbose'           Print out debug messages about
    %                               progress.
    %
    %           'useC'              Heavily speed up parsing of samples and
    %                               timestamps by relying on external
    %                               c-code (parseValueBlock.c,
    %                               parseDate.c). Requires compilation
    %                               using mex-interface or pre-compiled
    %                               binaries (.mexXXX files).
    %
    %           'ignoreSessionId'   Ignore session Ids but read all data in
    %                               File
    %
    %       EXAMPLES:
    %           
    %       Reading file information:
    %           >> fileinfo = AX3_readFile('foobar.cwa', 'info', 1, 'useC', 1)
    %               fileinfo = 
    %                 validPackets: [991997x3 double]
    %                        start: [1x1 struct]
    %                         stop: [1x1 struct]
    %                     deviceId: 33
    %                    sessionId: 0
    %           >> fileinfo.start
    %               ans = 
    %                   mtime: 7.3492e+05
    %                     str: '17-Feb-2012 12:56:25'
    %
    %       subsequent read of slice using pre-read packet info:
    %           >> data = AX3_readFile('foobar.cwa', ... 
    %               'validPackets', fileinfo.validPackets, ...
    %               'startTime', datenum('19-Feb-2012 00:00:00'), ... 
    %               'stopTime', datenum('20-Feb-2012 00:00:00'));
    %
    %           >> data = 
    %               validPackets: [73059x3 double]
    %                        ACC: [8766736x4 double]
    %                      LIGHT: [73059x2 double]
    %                       TEMP: [73059x2 double]       
    %
    %
    %   v0.2
    %       Nils Hammerla, 2012 <nils.hammerla@ncl.ac.uk>
    %
    
    % 
    % Copyright (c) 2012, Newcastle University, UK.
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
    
    %   TODO:
    %       - Implement binary search through packet-space capable of
    %         handling broken packages (remove need for pre-reading
    %         packages)
    %       - Add proper use of checksums
    %       - Write object oriented version capable of handling different
    %         versions of format (past and future)
    
    % parse arguments
    p = inputParser;
    
    % define optional arguments with default values
    addOptional(p,'info',0,@isnumeric);             % just get file-info?
    addOptional(p,'validPackets',0,@isnumeric);     % pre-read packet info
    addOptional(p,'startTime',-1,@isnumeric);       % start time (matlab)
    addOptional(p,'stopTime',-1,@isnumeric);        % stop time  (matlab)
    addOptional(p,'verbose',0,@isnumeric);          % print out progress
    addOptional(p,'useC',0,@isnumeric);             % use external c-code 
                                                    % for speed.
    addOptional(p,'ignoreSessionId',0,@isnumeric);             % use external c-code 
    addOptional(p,'modality',[1 1 1], @isnumeric);  % specify modality
                                                    % [ACC LIGHT TEMP]
    
    % parse varargin
    parse(p,varargin{:});
    
    % dispatch
    if p.Results.info > 0,
        data = readFileInfo(filename, p.Results);
    else
        data = readFile(filename, p.Results);
    end
    
    
end

function data = readFile(filename, options)
    % read data from file according to specified options
    
    % open file
    fid = fopen(filename);
    
    if options.verbose,
        fprintf('reading file: %s\n', filename);
    end


    % check if we have pre-read timestamps...
    if length(options.validPackets) == 1,
        % we don't, so read information
        data.validPackets = getValidPackets(fid, options);
    else
        % get the valid packet position from the given matrix
        data.validPackets = options.validPackets;
    end
    
    
    % get relevant parts of data if sliced reading enabled
    %
    
    % pre-set boundaries
    first = 1;
    last  = size(data.validPackets,1);
    
    % if startTime & stopTime given, get slice of data
    if options.startTime > 0 && options.stopTime > 0,
        if options.verbose > 0,
            fprintf('Finding relevant packets from %s to %s.\n',datestr(options.startTime),datestr(options.stopTime));
        end

        % find relevant parts of valid packets (second column =
        % matlab-timestamp)
        tmplist = find(data.validPackets(:,2) >= options.startTime & data.validPackets(:,2) <= options.stopTime);

        if isempty(tmplist),
            fprintf('No packets found in desired time-range! Aborting.\n');
        end

        first = tmplist(1);
        last = tmplist(end);
        
        clear tmplist;
    end
    
    % filter positions for period of interest
    data.validPackets = data.validPackets(first:last,:);
    
    % check if any packets are left...
    if isempty(data.validPackets),
        fprintf('no data after filtering for start and stop dates given.\n');
        return
    end
    
    % get the timestamps for each sample for interpolation later
    % First: get packets with positive timestamp offset (packet where
    % clocked flicked over to current time)
    packetIds = find(data.validPackets(:,3) > 0);
    
    % now, use timestamp offset to get (sample) position on which timestamp
    % occurred (120 samples per packet). TIME will be used for
    % interpolation of other timestamps
    TIME = [(packetIds*120)+double(data.validPackets(packetIds,3)) data.validPackets(packetIds,2)];
    % TIME_full just contains integer packet numbers for LIGHT and TEMP.
    TIME_full = [packetIds data.validPackets(packetIds,2)];
    
    % read data
    validIds = data.validPackets(:,1);
    
    % see what modalities to extract...
    if options.modality(1),
        % initialize variables
        data.ACC     = zeros(size(validIds,1)*120,4);
        ACC_tmp = zeros(size(validIds,1)*120,1,'uint32');
    end
    
    if options.modality(2),
        % initialize variables
        data.LIGHT   = zeros(size(validIds,1),2);
    end
    
    if options.modality(3),
        % initialize variables
        data.TEMP    = zeros(size(validIds,1),2);
    end
    
    if options.verbose,
        fprintf('reading data samples\n');
    end

    cnt = 1;
    
    % for each packet in valid packets, read samples
    for i=1:length(validIds),
        try % just to make sure
            
            
            if options.modality(2),
                % get light values
                fseek(fid,validIds(i)*512+18,-1);
                data.LIGHT(i,2) = fread(fid, 1, 'uint16',0,'ieee-le');
            end
            
            if options.modality(3) == 1,
                % get temperature values and convert to degree C
                fseek(fid,validIds(i)*512+20,-1);
                data.TEMP(i,2)  = (fread(fid, 1, 'uint16',0,'ieee-le') * 150 - 20500)/1000;
            end
            
            if options.modality(1) == 1,
                % read accelerometer values
                fseek(fid,validIds(i)*512+30,-1);
                % reads 120 unsigned integer (32 bit). Will be decoded
                % later on.
                ACC_tmp((i-1)*120+1:i*120) = fread(fid, 120, 'uint32',0,'ieee-le');
            end
            
            cnt = cnt + 1;
        end
    end

    if options.modality(1),
        % decode values for accelerometer

        if options.verbose > 0,
            fprintf('parsing samples\n');
        end
    
        % calls external c-code to unpack values (for speed pass on full
        % block instead of loop)
        
        if options.useC
            data.ACC(:,2:4) = parseValueBlock(ACC_tmp)' / 256;
        else
            data.ACC(:,2:4) = double(parseValueBlockML(ACC_tmp)) / 256;
        end
        
        clear ACC_tmp; % clear some memory
    end
    
    if options.verbose,
        fprintf('Calculating timestamps.\n');
    end
    
    if options.modality(1),
        % Interpolate the timestamps
        data.ACC = data.ACC(1:(cnt-2)*120,:); % drop last two packets
        % interpolate using TIME, linearly between known sample-based
        % timestamp locations.
        
        % clean TIME so that they are in strict monotonic order
        TIME = TIME([diff(TIME(:,1))~=0 ; true],:);
        
        data.ACC(:,1) = interp1(TIME(:,1),TIME(:,2),1:length(data.ACC),'linear',0);
        % clean up cases outside of known timestamps
        data.ACC = data.ACC(data.ACC(:,1)>0,:);
    end
    
    if options.modality(2) == 1,
        % interpolate time for light
        data.LIGHT(:,1) = interp1(TIME_full(:,1),TIME_full(:,2),1:length(data.LIGHT),'linear',0);
        data.LIGHT = data.LIGHT(data.LIGHT(:,1)>0,:);
    end
    
    if options.modality(3) == 1,
        % interpolate time for temperature
        data.TEMP(:,1) = interp1(TIME_full(:,1),TIME_full(:,2),1:length(data.TEMP),'linear',0);
        data.TEMP = data.TEMP(data.TEMP(:,1)>0,:);
    end
        
    % close the file, done.
    fclose(fid);
end

function validPackets = getValidPackets(fid, options)
    % read all the valid ids (accelerometer packets in file with correct
    % session id)
    
    % get length of file in bytes
    fseek(fid,0,1);                         % goto eof
    lengthBytes = ftell(fid);               % get position
    fseek(fid,0,-1);                        % rewind
    numPackets = floor(lengthBytes / 512);  % number of packets
    
    % Now reading headers and timestamps, etc. Refer to omapi.h for details
    % (OM_READER_DATA_PACKET)
    
    % get headers
    if options.verbose
        fprintf('reading headers\n');
    end
    packetHeaders = zeros(2,numPackets);
    packetHeaders(:) = fread(fid, numPackets*2, '2*uint8=>uint8',510,'ieee-le');
    
    % get timestamps
    if options.verbose
        fprintf('reading timestamps\n');
    end
	fseek(fid,14,-1);
	packetTimestamps = fread(fid, numPackets, 'uint32=>uint32',508,'ieee-le');
    
    % get offsets for timestamps
    if options.verbose
        fprintf('reading offsets\n');
    end
    fseek(fid,26,-1);
    packetOffsets = fread(fid,numPackets,'int16=>int16',510,'ieee-le');
    
    % get sessionIds
    if options.verbose
        fprintf('reading session-ids\n');
    end
    fseek(fid,6,-1);
    packetSessionIds = fread(fid, numPackets, 'uint32=>uint32',508,'ieee-le');
    
    % find valid packets
    validIds = find(sum(packetHeaders,1)==153);     % corresponds to specific string
    % all have to have the same (first) session id
    if options.ignoreSessionId == 0,
        validIds = validIds(packetSessionIds(validIds) == packetSessionIds(validIds(1)));
    end
        
    % return ids and timestamps
    if options.verbose
        fprintf('parsing timestamps\n');
    end
 	validPackets = zeros(length(validIds),3);
	validPackets(:,1) = validIds;
    
    for i=1:length(validIds),
		if options.useC
            validPackets(i,2) = datenum(parseDate(packetTimestamps(validIds(i))));
        else
            validPackets(i,2) = datenum(parseDateML(packetTimestamps(validIds(i))));
        end
        
        validPackets(i,3) = packetOffsets(validIds(i));
    end
    
end

function data = readFileInfo(filename, options)
    % read file general info

    % open file
    fid = fopen(filename);
    
    % read valid packet positions 
    data.validPackets = getValidPackets(fid, options);
    
    % get time (first and last valid packets)
    data.start.mtime = data.validPackets(1,2);
    data.start.str   = datestr(data.validPackets(1,2));
    data.stop.mtime  = data.validPackets(end,2);
    data.stop.str    = datestr(data.validPackets(end,2));
    
    % get deviceId from first packet
    fseek(fid, data.validPackets(1,1)*512+4,-1);
    data.deviceId = fread(fid, 1, 'uint16', 0, 'ieee-le');
    
    % get sessionId from first packet
    fseek(fid, data.validPackets(1,1)*512+6,-1);
    data.sessionId = fread(fid, 1, 'uint32', 0, 'ieee-le');
    
    % close the file
    fclose(fid);
end

%% matlab specific parsing functions (c-code much much quicker)

% Matlab version by Dan Jackson, 2012
function res = parseValueBlockML(value)
    % 32-bit packed value (The bits in bytes [3][2][1][0]: eezzzzzz zzzzyyyy yyyyyyxx xxxxxxxx, e = binary exponent, lsb on right)
    e = int32(2 .^ bitshift(value, -30));
    x = int32(bitand(         value      , 1023)); x = e .* (x - (int32(1024) .* int32(x >= 512)));
    y = int32(bitand(bitshift(value, -10), 1023)); y = e .* (y - (int32(1024) .* int32(y >= 512)));
    z = int32(bitand(bitshift(value, -20), 1023)); z = e .* (z - (int32(1024) .* int32(z >= 512)));
    res = [x y z];
end

% Matlab version by Dan Jackson, 2012
function datearr = parseDateML(t)
    datearr = zeros(1,6);

    % Y-M-D h:m:s
    % #define OM_DATETIME_YEAR(dateTime)    ((unsigned int)((unsigned char)(((dateTime) >> 26) & 0x3f)) + 2000) /**< Extract the year from a packed date/time value. @hideinitializer */
    % #define OM_DATETIME_MONTH(dateTime)   ((unsigned char)(((dateTime) >> 22) & 0x0f))  /**< Extract the month (1-12) from a packed date/time value. @hideinitializer */
    % #define OM_DATETIME_DAY(dateTime)     ((unsigned char)(((dateTime) >> 17) & 0x1f))  /**< Extract the day (1-31) from a packed date/time value. @hideinitializer */
    % #define OM_DATETIME_HOURS(dateTime)   ((unsigned char)(((dateTime) >> 12) & 0x1f))  /**< Extract the hours (0-23) from a packed date/time value. @hideinitializer */
    % #define OM_DATETIME_MINUTES(dateTime) ((unsigned char)(((dateTime) >>  6) & 0x3f))  /**< Extract the minutes (0-59) from a packed date/time value. @hideinitializer */
    % #define OM_DATETIME_SECONDS(dateTime) ((unsigned char)(((dateTime)      ) & 0x3f))  /**< Extract the seconds (0-59) from a packed date/time value. @hideinitializer */
   
    %%% 'idivide' version -- think this fixes potential problems with initial version
    datearr(1) = idivide(t, 67108864, 'floor') + 2000;
    datearr(2) = mod(idivide(t, 4194304, 'floor'), 16);
    datearr(3) = mod(idivide(t, 131072, 'floor'), 32);
    datearr(4) = mod(idivide(t, 4096, 'floor'), 32);
    datearr(5) = mod(idivide(t, 64, 'floor'), 64);
    datearr(6) = mod(t, 64);
  
end
