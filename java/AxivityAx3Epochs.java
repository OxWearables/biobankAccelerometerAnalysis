//BSD 2-Clause (c) 2014: A.Doherty (Oxford), D.Jackson, N.Hammerla (Newcastle)
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.math.RoundingMode;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.List;
import java.time.Duration;
import java.time.format.DateTimeFormatter;
import java.time.LocalDateTime;
import java.time.temporal.ChronoField;
import java.text.SimpleDateFormat;
import java.text.DecimalFormat;

/**
 * Calculates epoch summaries from an AX3 .CWA file.
 * Class/application can be called from the command line as follows:
 * java AxivityAx3Epochs inputFile.CWA 
 */
public class AxivityAx3Epochs
{
  
  private static DecimalFormat DF6 = new DecimalFormat("0.000000");
  private static DecimalFormat DF2 = new DecimalFormat("0.00");
  private static LocalDateTime SESSION_START = null;
  private static long START_OFFSET_NANOS = 0; 

  /**
   * Parse command line args, then call method to identify and write epochs.
   * @param   args  An argument string  passed in by ActivitySummary.py. Contains "param:value" pairs.
   */
  public static void main(String[] args) {
    //variables to store default parameter options
    String accFile = "";
    String[] functionParameters = new String[0];
    String outputFile = "";
    Boolean verbose = true;
    int epochPeriod = 5;
    String fmt = "yyyy-MM-dd HH:mm:ss.SSS";
    DateTimeFormatter timeFormat = DateTimeFormatter.ofPattern(fmt);
    double lowPassCut = 20;
    double highPassCut = 0.2;
    int sampleRate = 100;
    //create Filters necessary for later data processing
    LowpassFilter filter = new LowpassFilter(lowPassCut, sampleRate);
    //BandpassFilter filter = new BandpassFilter(highPassCut, lowPassCut, sampleRate);
    Boolean startEpochWholeMinute = false;
    Boolean startEpochWholeSecond = false;
    Boolean getStationaryBouts = false;
    double stationaryStd = 0.013;
    double[] swIntercept = new double[]{0.0, 0.0, 0.0};
    double[] swSlope = new double[]{1.0, 1.0, 1.0};
    double[] tempCoef = new double[]{0.0, 0.0, 0.0};
    double meanTemp = 0.0;
    int range = 8;
    Boolean rawOutput = false;
    DF6.setRoundingMode(RoundingMode.CEILING);
    DF2.setRoundingMode(RoundingMode.CEILING);
    if (args.length < 1) {
      String invalidInputMsg = "Invalid input, ";
      invalidInputMsg += "please enter at least 1 parameter, e.g.\n";
      invalidInputMsg += "java AxivityAx3Epochs inputFile.CWA";
      System.out.println(invalidInputMsg);
      System.exit(-1);
    } else if (args.length == 1) {
      //singe parameter needs to be accFile
      accFile = args[0]; 
      outputFile = accFile.split("\\.")[0] + "Epoch.csv";
    } else {
      //load accFile, and also copy functionParameters (args[1:])
      accFile = args[0];
      outputFile = accFile.split("\\.")[0] + "Epoch.csv";
      functionParameters = Arrays.copyOfRange(args, 1, args.length);

      //update default values by looping through available user parameters
      for (String param : functionParameters) {
        //individual_Parameters will look like "epoch_period:60"
        String funcName = param.split(":")[0];
        String funcParam = param.substring(param.indexOf(":") + 1);
        if (funcName.equals("outputFile")) {
          outputFile = funcParam;
        } else if (funcName.equals("verbose")) {
          verbose = Boolean.parseBoolean(funcParam.toLowerCase());
        } else if (funcName.equals("epochPeriod")) {
          epochPeriod = Integer.parseInt(funcParam);
        } else if (funcName.equals("timeFormat")) {
          timeFormat = DateTimeFormatter.ofPattern(funcParam);
        } else if (funcName.equals("filter")) {
          if (!Boolean.parseBoolean(funcParam.toLowerCase())) {
            filter = null;
          }
        } else if (funcName.equals("startEpochWholeMinute")) {
          startEpochWholeMinute = Boolean.parseBoolean(funcParam.toLowerCase());
        } else if (funcName.equals("startEpochWholeSecond")) {
          startEpochWholeSecond = Boolean.parseBoolean(funcParam.toLowerCase());
        } else if (funcName.equals("getStationaryBouts")) {
          getStationaryBouts = Boolean.parseBoolean(funcParam.toLowerCase());
          epochPeriod = 10;
        } else if (funcName.equals("stationaryStd")) {
          stationaryStd = Double.parseDouble(funcParam);
        } else if (funcName.equals("xIntercept")) {
          swIntercept[0] = Double.parseDouble(funcParam);
        } else if (funcName.equals("yIntercept")) {
          swIntercept[1] = Double.parseDouble(funcParam);
        } else if (funcName.equals("zIntercept")) {
          swIntercept[2] = Double.parseDouble(funcParam);
        } else if (funcName.equals("xSlope")) {
          swSlope[0] = Double.parseDouble(funcParam);
        } else if (funcName.equals("ySlope")) {
          swSlope[1] = Double.parseDouble(funcParam);
        } else if (funcName.equals("zSlope")) {
          swSlope[2] = Double.parseDouble(funcParam);
        } else if (funcName.equals("xTemp")) {
          tempCoef[0] = Double.parseDouble(funcParam);
        } else if (funcName.equals("yTemp")) {
          tempCoef[1] = Double.parseDouble(funcParam);
        } else if (funcName.equals("zTemp")) {
          tempCoef[2] = Double.parseDouble(funcParam);
        } else if (funcName.equals("meanTemp")) {
          meanTemp = Double.parseDouble(funcParam);
        } else if (funcName.equals("range")) {
          range = Integer.parseInt(funcParam);
        } else if (funcName.equals("rawOutput")) {
          rawOutput = Boolean.parseBoolean(funcParam.toLowerCase());
        }
      }
    }  
    
    String epochHeader = "Time,enmoTrunc,";
    if(getStationaryBouts){
      epochHeader += "xMean,yMean,zMean,";
    }
    epochHeader += "xRange,yRange,zRange,xStd,yStd,zStd,temp,samples,";
    epochHeader += "dataErrors,clipsBeforeCalibr,clipsAfterCalibr,rawSamples";
    
    //process file if input parameters are all ok
    if(accFile.toLowerCase().endsWith(".cwa")) {
        writeCwaEpochs(accFile, outputFile, epochHeader, verbose, epochPeriod,
                    timeFormat, startEpochWholeMinute, startEpochWholeSecond,
                    range, swIntercept, swSlope, tempCoef, meanTemp,
                    getStationaryBouts, stationaryStd, filter, rawOutput);  
    }
    else if(accFile.toLowerCase().endsWith(".bin")) {
        writeGeneaEpochs(accFile, outputFile, epochHeader, verbose, epochPeriod,
                    timeFormat, startEpochWholeMinute, startEpochWholeSecond,
                    range, swIntercept, swSlope, tempCoef, meanTemp,
                    getStationaryBouts, stationaryStd, filter, rawOutput);  
    }
    else {
        System.err.println("Unrecognised file format");
        System.exit(-1);
    } 
    
    //if no errors then will reach this
    System.exit(0);
  }


  /**
   * Read data block HEX values, store each raw reading, then continually test
   * if an epoch of data has been collected or not. Finally, write each epoch
   * to epochFileWriter. Method also updates and returns epochStartTime.
   * CWA format is described at:
   * https://code.google.com/p/openmovement/source/browse/downloads/AX3/AX3-CWA-Format.txt
   */
  private static void writeEpochSummary(
      BufferedWriter epochWriter,
      DateTimeFormatter timeFormat,
      LocalDateTime epochStartTime,
      int epochPeriod,
      double intendedSampleRate,
      List<Long> timeVals,
      List<Double> xVals,
      List<Double> yVals,
      List<Double> zVals,
      List<Double> temperatureVals,
      int range,
      int[] errCounter,
      double[] swIntercept,
      double[] swSlope,
      double[] tempCoef,
      double meanTemp,
      Boolean getStationaryBouts,
      double staticStd,
      LowpassFilter filter,
      PrintWriter rawWriter) {

    int[] clipsCounter = new int[]{0, 0}; //before, after (calibration)
    double x;
    double y;
    double z;
    for(int c=0; c<xVals.size(); c++){
      Boolean isClipped = false;
      x = xVals.get(c);
      y = yVals.get(c);
      z = zVals.get(c);
      double mcTemp = temperatureVals.get(c) - meanTemp; //mean centred temp
      //check if any clipping present, use ==range as it's clipped here
      if(x<=-range || x>=range || y<=-range || y>=range || z<=-range || z>=range){
        clipsCounter[0] += 1;
        isClipped = true;
      }

      //update values to software calibrated values
      x = swIntercept[0] + x*swSlope[0] + mcTemp*tempCoef[0];
      y = swIntercept[1] + y*swSlope[1] + mcTemp*tempCoef[1];
      z = swIntercept[2] + z*swSlope[2] + mcTemp*tempCoef[2];
      xVals.set(c, x);
      yVals.set(c, y);
      zVals.set(c, z);
      //check if any new clipping has happened
      //find crossing of range threshold so use < rather than ==
      if(x<-range || x>range || y<-range || y>range || z<-range || z>range){
        if (!isClipped) {
          clipsCounter[1] += 1;
        }
        //drag post calibration clipped values back to range limit
        if (x<-range || (isClipped && x<0)) {
          x = -range;
        }
        else if (x>range || (isClipped && x>0)) {
          x = range;
        }
        if (y<-range || (isClipped && y<0)) {
          y = -range;
        }
        else if (y>range || (isClipped && y>0)) {
          y = range;
        }
        if (z<-range || (isClipped && z<0)) {
          z = -range;
        }
        else if (z>range || (isClipped && z>0)) {
          z = range;
        }
      }
    }
    
    //resample values to epochSec * (intended) sampleRate
    long[] timeResampled = new long[epochPeriod * (int)intendedSampleRate];
    for(int c=0; c<timeResampled.length; c++){
      timeResampled[c] = timeVals.get(0) + (10*c);
    }
    double[] xResampled = new double[timeResampled.length];
    double[] yResampled = new double[timeResampled.length];
    double[] zResampled = new double[timeResampled.length];
    Resample.interpLinear(timeVals, xVals, yVals, zVals, timeResampled,
                          xResampled, yResampled, zResampled);
    
    //epoch variables
    String epochSummary = "";
    double accPA = 0;
    double xMean = 0;
    double yMean = 0;
    double zMean = 0;
    double xRange = 0;
    double yRange = 0;
    double zRange = 0;
    double xStd = 0;
    double yStd = 0;
    double zStd = 0;   

    //calculate raw x/y/z summary values
    xMean = mean(xResampled);
    yMean = mean(yResampled);
    zMean = mean(zResampled);
    xRange = range(xResampled);
    yRange = range(yResampled);
    zRange = range(zResampled);
    xStd = std(xResampled, xMean);
    yStd = std(yResampled, yMean);
    zStd = std(zResampled, zMean);

    //see if values have been abnormally stuck this epoch
    double stuckVal = 1.5;
    if (xStd==0 && (xMean<-stuckVal || xMean>stuckVal)) {
      errCounter[0] += 1;
    }
    if (yStd==0 && (yMean<-stuckVal || yMean>stuckVal)) {
      errCounter[0] += 1;
    }
    if (zStd==0 && (zMean<-stuckVal || zMean>stuckVal)) {
      errCounter[0] += 1;
    }
     
    //calculate summary vector magnitude based metrics
    List<Double> paVals = new ArrayList<Double>();
    if(!getStationaryBouts) {
      for(int c=0; c<xResampled.length; c++){
        x = xResampled[c];
        y = yResampled[c];
        z = zResampled[c];
        if (rawWriter!=null) {
          rawWriter.println(epochStartTime.plusNanos(START_OFFSET_NANOS+timeResampled[c]*1000000).format(timeFormat) + "," + x + "," + y + "," + z+","+meanTemp);
        }

        if(!Double.isNaN(x)) {
          double vm = getVectorMagnitude(x,y,z);
          paVals.add(vm-1);
        }
      }
      //filter AvgVm-1 values
      if (filter != null) {
        filter.filter(paVals);
      }
      //run abs() or trunc() on summary variables after filtering
      trunc(paVals);
      //calculate mean values for each outcome metric 
      accPA = mean(paVals);
    }
    //write summary values to file
    epochSummary = epochStartTime.plusNanos(START_OFFSET_NANOS).format(timeFormat);
    epochSummary += "," + DF6.format(accPA);
    if(getStationaryBouts){
      epochSummary += "," + DF6.format(xMean);
      epochSummary += "," + DF6.format(yMean);
      epochSummary += "," + DF6.format(zMean);
    }
    epochSummary += "," + DF6.format(xRange);
    epochSummary += "," + DF6.format(yRange);
    epochSummary += "," + DF6.format(zRange);
    epochSummary += "," + DF6.format(xStd);
    epochSummary += "," + DF6.format(yStd);
    epochSummary += "," + DF6.format(zStd);
    epochSummary += "," + DF2.format(mean(temperatureVals));
    epochSummary += "," + xResampled.length + "," + errCounter[0];
    epochSummary += "," + clipsCounter[0] + "," + clipsCounter[1];
    epochSummary += "," + timeVals.size(); 
    if(!getStationaryBouts || (xStd<staticStd && yStd<staticStd && zStd<staticStd)) {
      writeLine(epochWriter, epochSummary);    
    }
  }

  
  /**
   * Read Axivity CWA file, then call method to write epochs from raw data.
   * Epochs will be written to path "outputFile".
   */
  private static void writeCwaEpochs(
      String accFile,
      String outputFile,
      String epochHeader,
      Boolean verbose,
      int epochPeriod,
      DateTimeFormatter timeFormat,
      Boolean startEpochWholeMinute,
      Boolean startEpochWholeSecond,
      int range,
      double[] swIntercept,
      double[] swSlope,
      double[] tempCoef,
      double meanTemp,
      Boolean getStationaryBouts,
      double staticStd,
      LowpassFilter filter,
      Boolean rawOutput) {
    //epoch creation support variables
    LocalDateTime epochStartTime = null;
    List<Long> timeVals = new ArrayList<Long>();
    List<Double> xVals = new ArrayList<Double>();
    List<Double> yVals = new ArrayList<Double>();
    List<Double> zVals = new ArrayList<Double>();
    List<Double> temperatureVals = new ArrayList<Double>();
    int[] errCounter = new int[]{0}; //store val if updated in other method
    // Inter-block timstamp tracking
    LocalDateTime[] lastBlockTime = { null };
    int[] lastBlockTimeIndex = { 0 };
    
    //data block support variables
    String header = "";      
    //file read/write objects
    FileChannel rawAccReader = null;
    BufferedWriter epochFileWriter = null;
    PrintWriter rawWriter = null;
    int bufSize = 512;
    ByteBuffer buf = ByteBuffer.allocate(bufSize);    
    try {
      rawAccReader = new FileInputStream(accFile).getChannel();
      epochFileWriter = new BufferedWriter(new FileWriter(outputFile));
      if (rawOutput) {
        rawWriter = new PrintWriter(new BufferedWriter(new FileWriter(outputFile+"_raw.csv")));
        rawWriter.println("time,x,y,z,meanTemp");
      }

      //now read every page in CWA file
      int pageCount = 0;
      long memSizePages = rawAccReader.size()/bufSize;
      boolean USE_PRECISE_TIME = true; //true uses block fractional time and
                                        //interpolates timestamp between blocks.
      while(rawAccReader.read(buf) != -1) {
        buf.flip();
        buf.order(ByteOrder.LITTLE_ENDIAN);
        header = (char)buf.get() + "";
        header += (char)buf.get() + "";
        if(header.equals("MD")) {
          //Read first page (& data-block) to get time, temp, measureFreq
          //start-epoch values
          try {
            SESSION_START = cwaHeaderLoggingStartTime(buf);
            System.out.println("Session start:" + SESSION_START);
          }
          catch (Exception e) {
            System.err.println("No preset start time");
          }
          writeLine(epochFileWriter, epochHeader);
        }
        else if(header.equals("AX")) {
          //read each individual page block, and process epochs...
          try{
            //read block header items
            long blockTimestamp = getUnsignedInt(buf,14);
            int light = getUnsignedShort(buf,18);
            double temperature = (getUnsignedShort(buf,20)*150.0- 20500) / 1000;
            short rateCode = (short)(buf.get(24) & 0xff);
            short numAxesBPS = (short)(buf.get(25) & 0xff);
            int sampleCount = getUnsignedShort(buf, 28);
            short timestampOffset = 0;
            double sampleFreq = 0;
            int fractional = 0; // 1/65536th of a second fractions

            //check not very old file as pos 26=freq rather than
            // timestamp offset
            if (rateCode != 0) {
              timestampOffset = buf.getShort(26); //timestamp offset ok
              //if fractional offset, then timestamp offset was artificially
              //modified for backwards-compatibility ... therefore undo this...
              int oldDeviceId = getUnsignedShort(buf, 4);
              if ((oldDeviceId & 0x8000) != 0) {
                sampleFreq = 3200.0 / (1 << (15 - (rateCode & 15)));
                if (USE_PRECISE_TIME) {
                  // Need to undo backwards-compatible shim:
                  // Take into account how many whole samples
                  // the fractional part of timestamp 
                  // accounts for:  
                  // relativeOffset = fifoLength - (short)(((unsigned long)timeFractional * AccelFrequency()) >> 16);
                  //               nearest whole sample
                  //      whole-sec   | /fifo-pos@time
                  //       |        |/
                  // [0][1][2][3][4][5][6][7][8][9]
                  // use 15-bits as 16-bit fractional time
                  fractional = ((oldDeviceId & 0x7fff) << 1);
                  //frequency is truncated to int in firmware
                  timestampOffset += ((fractional * (int)sampleFreq) >> 16);
                }
              }
            }
            else {
              sampleFreq = buf.getShort(26);
              //very old format, where pos26 = freq
            }
            
            //calculate num bytes per sample...
            byte bytesPerSample = 4;
            int NUM_AXES_PER_SAMPLE = 3;
            if ((numAxesBPS & 0x0f) == 2) {
              bytesPerSample = 6; // 3*16-bit
            }
            else if ((numAxesBPS & 0x0f) == 0) {
              bytesPerSample = 4; // 3*10-bit + 2
            }
           
            // Limit values
            int maxSamples = 480 / bytesPerSample; // 80 or 120 samples/block
            if (sampleCount > maxSamples) {
              sampleCount = maxSamples;
            }
            if (sampleFreq <= 0) {
              sampleFreq = 1;
            }
            
            // determine time for indexed sample within block
            LocalDateTime blockTime = getCwaTimestamp( (int)blockTimestamp,
                                                        fractional);    
            // first & last sample. Actually, last = first sample in next block
            LocalDateTime firstSampleTime, lastSampleTime;
            // if no interval between times (or interval too large)
            long spanToSample = 0;
            if (lastBlockTime[0] != null) {
              spanToSample = Duration.between(lastBlockTime[0],
                                              blockTime).toNanos();
            }
            if (!USE_PRECISE_TIME || lastBlockTime[0] == null ||
                timestampOffset <= lastBlockTimeIndex[0] || 
                spanToSample <= 0 ||
                spanToSample > 1000000000.0 * 2 * maxSamples / sampleFreq
                ) {
              float offsetStart = (float)-timestampOffset / (float)sampleFreq;
              firstSampleTime = blockTime.plusNanos(secs2Nanos(offsetStart));
              lastSampleTime = firstSampleTime.plusNanos(
                                          secs2Nanos(sampleCount / sampleFreq));
            } else {
              double gap = (double)spanToSample 
                            / (-lastBlockTimeIndex[0] + timestampOffset);
              firstSampleTime = lastBlockTime[0].plusNanos(
                                          (long)(-lastBlockTimeIndex[0] * gap));
              lastSampleTime = lastBlockTime[0].plusNanos(
                                  (long)(
                                      (-lastBlockTimeIndex[0] + sampleCount) * gap
                                      )
                                  );
            }

            // Last block time
            lastBlockTime[0] = blockTime;
            //Advance last block time index for next block
            lastBlockTimeIndex[0] = timestampOffset - sampleCount;
            // Overall span
            long spanNanos = Duration.between(firstSampleTime,
                                                lastSampleTime).toNanos();
            
            //set target epoch start time of very first block
            if(epochStartTime==null) {
              epochStartTime = firstSampleTime;
              //if set, clamp session to intended log start time
              if(SESSION_START!=null) {
                START_OFFSET_NANOS = Duration.between(epochStartTime,
                                                      SESSION_START).toNanos();
                //check block time and session start time are within 10secs
                long clampLimitNanos = secs2Nanos(15);
                if( START_OFFSET_NANOS > clampLimitNanos || 
                    START_OFFSET_NANOS < -clampLimitNanos ){
                  START_OFFSET_NANOS = 0;
                  System.out.println("Can't clamp to logging start time");
                }
              }
            }

            //raw reading values
            long value = 0; // x/y/z vals
            short xRaw = 0;
            short yRaw = 0;
            short zRaw = 0;
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            
            //loop through each line in data block and check if it is last in
            //epoch then write epoch summary to file
            //an epoch will have a start+end time, and fixed duration      
            int currentPeriod;
            for (int i = 0; i<sampleCount; i++) {
              //Calculate each sample's time, not successively adding so that
              //we don't accumulate any errors
              if (USE_PRECISE_TIME) {
                blockTime = firstSampleTime.plusNanos
                                    ( (long)(i * (double)spanNanos / sampleCount) 
                                    );
              }
              else if (i == 0) {
                blockTime = firstSampleTime; //emulate original behaviour
              }
              
              if (bytesPerSample == 4) {
                try {
                  value = getUnsignedInt(buf, 30 +4*i);
                } 
                catch (Exception excep) {
                  errCounter[0] += 1;
                  System.err.println("xyz reading err: " + excep.toString());
                  break; //rest of block/page may be corrupted
                }
                // Sign-extend 10-bit values, adjust for exponents
                xRaw = (short)( (short)(0xffffffc0 & (value <<  6))
                                >> (6 - ((value >> 30) & 0x03)) );
                yRaw = (short)( (short)(0xffffffc0 & (value >>  4))
                                >> (6 - ((value >> 30) & 0x03)) );
                zRaw = (short)( (short)(0xffffffc0 & (value >>  14))
                                >> (6 - ((value >> 30) & 0x03)) );
              }
              else if (bytesPerSample == 6) {
                try {
                  errCounter[0] += 1;
                  xRaw = buf.getShort(30 + 2 * NUM_AXES_PER_SAMPLE * i + 0);
                  yRaw = buf.getShort(30 + 2 * NUM_AXES_PER_SAMPLE * i + 2);
                  zRaw = buf.getShort(30 + 2 * NUM_AXES_PER_SAMPLE * i + 4);
                }
                catch (Exception excep) {
                  System.err.println("xyz read err: " + excep.toString());
                  break; //rest of block/page may be corrupted
                }
              }
              else {
                xRaw = 0;
                yRaw = 0;
                zRaw = 0;
              }      
              x = xRaw / 256.0;
              y = yRaw / 256.0;
              z = zRaw / 256.0;
              currentPeriod = (int)Duration.between(epochStartTime,
                                                    blockTime).getSeconds();
              
              //check for an interrupt i.e. where break in vals>2*epochPeriod
              if (currentPeriod >= epochPeriod*2) {
                int epochDiff = currentPeriod/epochPeriod;
                epochStartTime = epochStartTime.plusSeconds(
                                                        epochPeriod*epochDiff);
                //and update how far we are into the new epoch...
                currentPeriod = (int) (
                                (blockTime.get(ChronoField.MILLI_OF_SECOND)
                                 - epochStartTime.get(ChronoField.MILLI_OF_SECOND)
                                 ) / 1000 );
              }
              
              //check we have collected enough values to form an epoch
              if (currentPeriod >= epochPeriod) {
                writeEpochSummary(epochFileWriter, timeFormat, epochStartTime,
                                  epochPeriod, sampleFreq, timeVals, xVals,
                                  yVals, zVals, temperatureVals, range,
                                  errCounter, swIntercept, swSlope, tempCoef,
                                  meanTemp, getStationaryBouts, staticStd,
                                  filter, rawWriter);
                //reset target start time and reset arrays for next epoch
                epochStartTime = epochStartTime.plusSeconds(epochPeriod);
                timeVals.clear();
                xVals.clear();
                yVals.clear();
                zVals.clear();
                temperatureVals.clear();
                errCounter[0] = 0;
              }
              
              //store axes + vector magnitude vals for every reading
              timeVals.add(Duration.between(epochStartTime,
                                            blockTime).toMillis());
              xVals.add(x);
              yVals.add(y);
              zVals.add(z);
              temperatureVals.add(temperature);
              if (!USE_PRECISE_TIME) {
                // Moved this to recalculate at top (rather than potentially
                // accumulate slight errors with repeated addition)
                blockTime = blockTime.plusNanos( secs2Nanos(1.0 / sampleFreq) );
              }
            }
          }
          catch(Exception excep){
            excep.printStackTrace(System.err);
            System.err.println("block err @ " + epochStartTime.toString()
                                + ": " + excep.toString() );
          }
        }
        buf.clear();
        //option to provide status update to user...
        pageCount++;
        if(verbose && pageCount % 10000 == 0) {
          System.out.print((pageCount*100/memSizePages) + "%\t");
        }
      } 
      rawAccReader.close();
      epochFileWriter.close();
    }
    catch (Exception excep) {
      excep.printStackTrace(System.err);
      System.err.println("error reading/writing file " + outputFile
                          + ": " + excep.toString() );
      System.exit(-2);
    }
  }

  //Parse HEX values, CWA format is described at:
  //https://code.google.com/p/openmovement/source/browse/downloads/AX3/AX3-CWA-Format.txt
  private static LocalDateTime getCwaTimestamp(
      int cwaTimestamp,
      int fractional) {
    LocalDateTime tStamp;
    int year = (int)((cwaTimestamp >> 26) & 0x3f) + 2000;
    int month = (int)((cwaTimestamp >> 22) & 0x0f);
    int day = (int)((cwaTimestamp >> 17) & 0x1f);
    int hours = (int)((cwaTimestamp >> 12) & 0x1f);
    int mins = (int)((cwaTimestamp >>  6) & 0x3f);
    int secs = (int)((cwaTimestamp    ) & 0x3f);
    tStamp = LocalDateTime.of(year, month, day, hours, mins, secs);
    // add 1/65536th fractions of a second
    tStamp = tStamp.plusNanos(secs2Nanos(fractional / 65536.0));
    return tStamp;
  }      
  
  private static LocalDateTime cwaHeaderLoggingStartTime(ByteBuffer buf) {
    long delayedLoggingStartTime = getUnsignedInt(buf,13);
    return getCwaTimestamp((int)delayedLoggingStartTime, 0);
  }

  //credit for next 2 methods goes to:
  //http://stackoverflow.com/questions/9883472/is-it-possiable-to-have-an-unsigned-bytebuffer-in-java
  private static long getUnsignedInt(ByteBuffer bb, int position) {
    return ((long) bb.getInt(position) & 0xffffffffL);
  }

  private static int getUnsignedShort(ByteBuffer bb, int position) {
    return (bb.getShort(position) & 0xffff);
  }
 

  /**
   * Read GENEA bin file pages, then call method to write epochs from raw data.
   * Epochs will be written to path "outputFile".
   */
  private static void writeGeneaEpochs(
      String accFile,
      String outputFile,
      String epochHeader,
      Boolean verbose,
      int epochPeriod,
      DateTimeFormatter timeFormat,
      Boolean startEpochWholeMinute,
      Boolean startEpochWholeSecond,
      int range,
      double[] swIntercept,
      double[] swSlope,
      double[] tempCoef,
      double meanTemp,
      Boolean getStationaryBouts,
      double staticStd,
      LowpassFilter filter,
      Boolean rawOutput) {

    int fileHeaderSize = 59;
    int linesToAxesCalibration = 47;
    int pageHeaderSize = 9;
    //epoch creation support variables
    LocalDateTime epochStartTime = null;
    List<Long> timeVals = new ArrayList<Long>();
    List<Double> xVals = new ArrayList<Double>();
    List<Double> yVals = new ArrayList<Double>();
    List<Double> zVals = new ArrayList<Double>();
    List<Double> temperatureVals = new ArrayList<Double>();
    int[] errCounter = new int[]{0}; //store val if updated in other method
    
    //file read/write objects
    BufferedReader rawAccReader = null;
    BufferedWriter epochFileWriter = null;
    PrintWriter rawWriter = null;
    try {
      rawAccReader = new BufferedReader(new FileReader(accFile));
      epochFileWriter = new BufferedWriter(new FileWriter(outputFile));
      if (rawOutput) {
        rawWriter = new PrintWriter(new BufferedWriter(new FileWriter(outputFile+"_raw.csv")));
        rawWriter.println("time,x,y,z,meanTemp");
      }
      //Read header to determine mfrGain and mfrOffset values
      double[] mfrGain = new double[3];
      int[] mfrOffset = new int[3];
      int memSizePages = 0; //memory size in pages
      memSizePages = parseBinFileHeader(rawAccReader, fileHeaderSize,
                                        linesToAxesCalibration, mfrGain,
                                        mfrOffset);    
      writeLine(epochFileWriter, epochHeader);
      
      int pageCount=1;
      String page;
      String header = "";
      LocalDateTime blockTime = LocalDateTime.of(1999, 1, 1, 1, 1, 1);
      double temperature = 0.0;
      double sampleFreq = 0.0;
      String dataBlock = "";
      String timeFmtStr = "yyyy-MM-dd HH:mm:ss:SSS"; 
      DateTimeFormatter timeFmt = DateTimeFormatter.ofPattern(timeFmtStr);
      while ((page = readLine(rawAccReader)) != null) {  
        //header: "Recorded Data" (0), serialCode (1), seq num (2),
        //    pageTime (3), unassigned (4), temp (5), batteryVolt (6),
        //    deviceStatus (7), sampleFreq (8),
        //Then: dataBlock (9)
        //line "page = readLine(..." above will read 1st header line (c=0)
        for(int c = 1; c < pageHeaderSize; c++) {
          try
          {
            header = readLine(rawAccReader); 
            if (c == 3) {
              blockTime = LocalDateTime.parse(header.split("Time:")[1], timeFmt);
            }
            else if (c == 5) {
              temperature = Double.parseDouble(header.split(":")[1]);
            }
            else if (c == 8 && epochStartTime == null) {
              sampleFreq = Double.parseDouble(header.split(":")[1]);              
            }
          }
          catch(Exception excep)
          {
            System.err.println(excep.toString());
            continue; //to keep reading sequence correct
          }
        }        
        //now process hex dataBlock
        dataBlock = readLine(rawAccReader);
        
        //set target epoch start time of very first block
        if(epochStartTime==null) {
          epochStartTime = blockTime;
        }

        //raw reading values
        int hexPosition = 0;
        int xRaw = 0;
        int yRaw = 0;
        int zRaw = 0;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        
        //loop through each reading in data block and check if it is last in
        //epoch then write epoch summary to file
        //an epoch will have a start+end time, and fixed duration      
        int currentPeriod;
        while (hexPosition < dataBlock.length()) {
          try
          {
            xRaw = getSignedIntFromHex(dataBlock, hexPosition, 3);
            yRaw = getSignedIntFromHex(dataBlock, hexPosition+3, 3);
            zRaw = getSignedIntFromHex(dataBlock, hexPosition+6, 3);
          }
          catch (Exception excep)
          {
            errCounter[0] += 1;
            System.err.println("block err @ " + epochStartTime.toString()
                                + ": " + excep.toString() );
            break; //rest of block/page could be corrupted
          }
          //todo *** read in light[36:46] (10 bits to signed int) and button[47] (bool) values...

          //update values to calibrated measure (taken from GENEActiv manual)
          x = (xRaw*100 - mfrOffset[0]) / mfrGain[0];
          y = (yRaw*100 - mfrOffset[1]) / mfrGain[1];
          z = (zRaw*100 - mfrOffset[2]) / mfrGain[2]; //todo *** is it ok to divide by int here?!!!

          currentPeriod = (int)Duration.between(epochStartTime,
                                                    blockTime).getSeconds();
              
          //check for an interrupt i.e. where break in vals>2*epochPeriod
          if (currentPeriod >= epochPeriod*2) {
            int epochDiff = currentPeriod/epochPeriod;
            epochStartTime = epochStartTime.plusSeconds(
                                                    epochPeriod*epochDiff);
            //and update how far we are into the new epoch...
            currentPeriod = (int) (
                            (blockTime.get(ChronoField.MILLI_OF_SECOND)
                             - epochStartTime.get(ChronoField.MILLI_OF_SECOND)
                             ) / 1000 );
          }
              
          //check we have collected enough values to form an epoch
          if (currentPeriod >= epochPeriod) {
            writeEpochSummary(epochFileWriter, timeFormat, epochStartTime,
                              epochPeriod, sampleFreq, timeVals, xVals, yVals,
                              zVals, temperatureVals, range, errCounter,
                              swIntercept, swSlope, tempCoef, meanTemp,
                              getStationaryBouts, staticStd, filter, rawWriter);
            //reset target start time and reset arrays for next epoch
            epochStartTime = epochStartTime.plusSeconds(epochPeriod);
            timeVals.clear();
            xVals.clear();
            yVals.clear();
            zVals.clear();
            temperatureVals.clear();
            errCounter[0] = 0;
          }
              
          //store axes + vector magnitude vals for every reading
          timeVals.add(Duration.between(epochStartTime,
                                        blockTime).toMillis());
          xVals.add(x);
          yVals.add(y);
          zVals.add(z);
          temperatureVals.add(temperature);
          //System.out.println(blockTime.format(timeFormat) + "," + x + "," + y + "," + z);
          hexPosition += 12; 
          blockTime = blockTime.plusNanos( secs2Nanos(1.0 / sampleFreq) );
        }
        //option to provide status update to user...
        pageCount++;
        if(verbose && pageCount % 10000 == 0) {
          System.out.print((pageCount*100/memSizePages) + "%\t");
        }
      } 
      rawAccReader.close();
      epochFileWriter.close();
    }
    catch (Exception excep) {
      excep.printStackTrace(System.err);
      System.err.println("error reading/writing file " + outputFile
                          + ": " + excep.toString() );
      System.exit(-2);
    }
  }
  
  /**
   * Replicates bin file header to epoch file, also calculates and returns
   * x/y/z gain/offset values along with number of pages of data in file
   * bin format described in GENEActiv manual ("Decoding .bin files", pg.27) 
   * http://www.geneactiv.org/wp-content/uploads/2014/03/geneactiv_instruction_manual_v1.2.pdf
   */
  private static int parseBinFileHeader(
      BufferedReader reader,
      int fileHeaderSize,
      int linesToAxesCalibration,
      double[] gainVals,
      int[] offsetVals)
  {
    //read first c lines in bin file to writer
    for (int c = 0; c < linesToAxesCalibration; c++) {
      readLine(reader);
    }
    //read axes calibration lines for gain and offset values
    //data like -> x gain:25548 \n x offset:574 ... Volts:300 \n Lux:800
    gainVals[0] = Double.parseDouble(readLine(reader).split(":")[1]); //xGain
    offsetVals[0] = Integer.parseInt(readLine(reader).split(":")[1]); //xOffset
    gainVals[1] = Double.parseDouble(readLine(reader).split(":")[1]); //y
    offsetVals[1] = Integer.parseInt(readLine(reader).split(":")[1]); //y
    gainVals[2] = Double.parseDouble(readLine(reader).split(":")[1]); //z
    offsetVals[2] = Integer.parseInt(readLine(reader).split(":")[1]); //z
    int volts = Integer.parseInt(readLine(reader).split(":")[1]); //volts
    int lux = Integer.parseInt(readLine(reader).split(":")[1]); //lux
    readLine(reader); //9 blank
    readLine(reader); //10 memory status header
    int memorySizePages = Integer.parseInt(readLine(reader).split(":")[1]); //11

    //ignore remaining header lines in bin file
    for (int c = 0; c < fileHeaderSize-linesToAxesCalibration-11; c ++) {
      readLine(reader);
    }      
    return memorySizePages;
  }
    
  private static int getSignedIntFromHex(String dataBlock,
      int startPos,
      int length)
  {
    //input hex base is 16
    int rawVal = Integer.parseInt(dataBlock.substring(startPos,startPos+length),16);
    int unsignedLimit = 4096; //2^[length*4] #i.e. 3 hexBytes (12 bits) limit = 4096
    int signedLimit = 2048; //2^[length*(4-1)] #i.e. 3 hexBytes - 1 bit (11 bits) limit = 2048
    if (rawVal > signedLimit) {
      rawVal = rawVal - unsignedLimit;
    }
    return rawVal;
  }
  
  
  private static double getVectorMagnitude(double x, double y, double z) {
    return Math.sqrt(x*x + y*y + z*z);
  }

  private static void abs(List<Double> vals) {
    for(int c=0; c<vals.size(); c++) {
      vals.set(c, Math.abs(vals.get(c)));
    }
  }
  
  private static void trunc(List<Double> vals) {
    double tmp;
    for(int c=0; c<vals.size(); c++) {
      tmp = vals.get(c);
      if(tmp < 0){
        tmp = 0;
      }
      vals.set(c, tmp);
    }
  }

  private static double sum(double[] vals) {
    if(vals.length==0) {
      return Double.NaN;
    }
    double sum = 0;
    for(int c=0; c<vals.length; c++) {
      if(!Double.isNaN(vals[c])) {
        sum += vals[c];
      }
    }
    return sum;
  }
  
  private static double mean(double[] vals) {
    if(vals.length==0) {
      return Double.NaN;
    }
    return sum(vals) / (double)vals.length;
  }
  
  private static double mean(List<Double> vals) {
    if(vals.size()==0) {
      return Double.NaN;
    }
    return sum(vals) / (double)vals.size();
  }
  
  private static double sum(List<Double> vals) {
    if(vals.size()==0) {
      return Double.NaN;
    }
    double sum = 0;
    for(int c=0; c<vals.size(); c++) {
      sum += vals.get(c);
    }
    return sum;
  }
    
  private static double range(double[] vals) {
    if(vals.length==0) {
      return Double.NaN;
    }
    double min = Double.MAX_VALUE;
    double max = Double.MIN_VALUE;
    for(int c=0; c<vals.length; c++) {
      if (vals[c] < min) {
        min = vals[c];
      }
      if (vals[c] > max) {
        max = vals[c];
      }
    }
    return max - min;
  }   

  private static double std(double[] vals, double mean) {
    if(vals.length==0) {
      return Double.NaN;
    }
    double var = 0; //variance
    double len = vals.length*1.0; //length
    for(int c=0; c<vals.length; c++) {
      if(!Double.isNaN(vals[c])) {
        var += ((vals[c] - mean) * (vals[c] - mean)) / len;
      }
    }
    return Math.sqrt(var);
  }

  private static long secs2Nanos(double num){
    return (long)(TimeUnit.SECONDS.toNanos(1)*num);
  }
  
  private static String readLine(BufferedReader fReader)
  {
    String line = "";
    try {
      line = fReader.readLine();
    }
    catch (Exception excep) {
      System.err.println(excep.toString());
    }
    return line;
  }
  
  private static void writeLine(BufferedWriter fileWriter, String line) {
    try {
      fileWriter.write(line + "\n");
    }
    catch (Exception excep) {
      System.err.println("line write error: " + excep.toString());
    }
  }
    
}
