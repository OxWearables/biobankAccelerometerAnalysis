import java.io.*;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.InputStream.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Date;
import java.util.GregorianCalendar;
import java.util.List;
import java.util.Map;
import java.text.SimpleDateFormat;

/**
 * Calculates epoch summaries from an AX3 .WAV file.
 * Class/application can be called from the command line as follows:
 * java AxivityAx3WavEpochs inputFile.wav 
 */
public class AxivityAx3WavEpochs 
{
    /**
     * Parse command line args, then call method to identify & write epochs.
     */
    public static void main(String[] args)
    {
        //variables to store default parameter options
        String accFile = "";
        String[] functionParameters = new String[0];
        String outputFile = "";
        int epochPeriod = 60;
        SimpleDateFormat timeFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.S");
        BandpassFilter filter = new BandpassFilter(0.50, 15, 100);
        Boolean startEpochWholeMinute = false;
        Boolean startEpochWholeSecond = false;
        Boolean zAxisTempCompensation = true;
        Boolean interpolateSample = true;
        if (args.length < 1) {
            String invalidInputMsg = "Invalid input, ";
            invalidInputMsg += "please enter at least 1 parameter, e.g.\n";
            invalidInputMsg += "java AxivityAx3WavEpochs inputFile.wav";
            System.out.println(invalidInputMsg);
            System.exit(0);
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
            for (String individualParam : functionParameters) {
                //individual_Parameters will look like "epoch_period:60"
                String funcName = individualParam.split(":")[0];
                String funcParam = individualParam.split(":")[1];
                if (funcName.equals("outputFile")) {
                    outputFile = funcParam;
                } else if (funcName.equals("epochPeriod")) {
                    epochPeriod = Integer.parseInt(funcParam);
                } else if (funcName.equals("timeFormat")) {
                    timeFormat = new SimpleDateFormat(funcParam);
                } else if (funcName.equals("filter")) {
                    if (!Boolean.parseBoolean(funcParam.toLowerCase())) {
                            filter = null; //i.e. we don't want default filter
                        }
                } else if (funcName.equals("startEpochWholeMinute")) {
                    startEpochWholeMinute = Boolean.parseBoolean(
                            funcParam.toLowerCase());
                } else if (funcName.equals("startEpochWholeSecond")) {
                    startEpochWholeSecond = Boolean.parseBoolean(
                            funcParam.toLowerCase());
                } else if (funcName.equals("zAxisTempCompensation")) {
                    zAxisTempCompensation = Boolean.parseBoolean(
                            funcParam.toLowerCase());
                }
            }
        }

        Calendar fileStartTime = getHeaderInfo(accFile, timeFormat);
        //process file if input parameters are all ok
        writeWavEpochs(accFile, outputFile, fileStartTime, epochPeriod, 100,
                timeFormat, startEpochWholeMinute, startEpochWholeSecond,
                filter);
    }
    
    private static void writeWavEpochs(
            String accFile,
            String outputFile,
            Calendar epochStartTime,
            int epochPeriod,
            int sampleRate,
            SimpleDateFormat timeFormat,
            Boolean startEpochWholeMinute,
            Boolean startEpochWholeSecond,
            BandpassFilter filter){
        AudioInputStream inputStream = null;
        BufferedWriter epochFileWriter = null;
        
        //data block support variables
        String header = "";        
        //epoch creation support variables
        List<Date> epochDatetimeArray = new ArrayList<Date>();
        List<Double> epochAvgVmVals = new ArrayList<Double>();
        List<Double> xVals = new ArrayList<Double>();
        List<Double> yVals = new ArrayList<Double>();
        List<Double> zVals = new ArrayList<Double>();
        String epochSummary = "";
        String epochHeader = "Timestamp,AvgVm,Xmean,Ymean,Zmean,Xrange,";
        epochHeader += "Yrange,Zrange,Xstd,Ystd,Zstd,Temp,Samples";
        //epoch variables
        double avgVm = 0;
        double xMean = 0;
        double yMean = 0;
        double zMean = 0;
        double xRange = 0;
        double yRange = 0;
        double zRange = 0;
        double xStd = 0;
        double yStd = 0;
        double zStd = 0;
        //raw reading values
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        int temperature = 0;
            
        try {
            inputStream = AudioSystem.getAudioInputStream(new File(accFile));
            epochFileWriter = new BufferedWriter(new FileWriter(outputFile));
            writeLine(epochFileWriter, epochHeader);
            int bytesPerFrame = inputStream.getFormat().getFrameSize();
            if (bytesPerFrame == AudioSystem.NOT_SPECIFIED) {
                // some audio formats may have unspecified frame size
                bytesPerFrame = 1; //read any amount of bytes
                System.out.println("Data channel appears invalid");
                System.exit(0);
            }
            //read in whole epoch (6000 samples) as a buffer
            int numBytes = bytesPerFrame * epochPeriod * sampleRate;
            byte[] epochBuf = new byte[numBytes];
            int numBytesRead = 0;
            //Loop through each epoch in File, i.e. read epochBuf bytes each time
            while ((numBytesRead = inputStream.read(epochBuf)) != -1) {
                // Loop through frames to read epoch values
                for (int s=0 ; s<epochBuf.length ; s+=bytesPerFrame) {
                    //read channels 1, 2, & 3 (but not 4)
                    x = (epochBuf[s+1] << 8 | (epochBuf[s]&0xFF)) /4096.0; //c1
                    y = (epochBuf[s+3] << 8 | (epochBuf[s+2]&0xFF)) /4096.0; //c2
                    z = (epochBuf[s+5] << 8 | (epochBuf[s+4]&0xFF)) /4096.0; //c3
                    //If valid, record x/y/z vals (i.e. not all zeros)
                    if(x!=0 && y!=0 && z!=0) {
                        xVals.add(x);
                        yVals.add(y);
                        zVals.add(z);
                        epochAvgVmVals.add(getVectorMagnitude(x,y,z));
                    }
                }
                //Now calculate epoch summary values...
                //band-pass filter AvgVm-1 values
                if (filter != null) {
                    filter.filter(epochAvgVmVals);
                }
                //take abs(AvgVm-1) vals which must be done after filtering
                abs(epochAvgVmVals);
                //calculate epoch summary values
                avgVm = mean(epochAvgVmVals);
                xMean = mean(xVals);
                yMean = mean(yVals);
                zMean = mean(zVals);
                xRange = range(xVals);
                yRange = range(yVals);
                zRange = range(zVals);
                xStd = std(xVals, xMean);
                yStd = std(yVals, yMean);
                zStd = std(zVals, zMean);
                //write summary values to file
                epochSummary = timeFormat.format(epochStartTime.getTime());
                epochSummary += "," + avgVm;
                epochSummary += "," + xMean + "," + yMean + "," + zMean;
                epochSummary += "," + xRange + "," + yRange + "," + zRange;
                epochSummary += "," + xStd + "," + yStd + "," + zStd;
                epochSummary += "," + temperature + "," + xVals.size();
                writeLine(epochFileWriter, epochSummary);

                //reset start time and arrays for next epoch
                epochStartTime.add(Calendar.SECOND, epochPeriod);
                xVals.clear();
                yVals.clear();
                zVals.clear();
                epochAvgVmVals.clear();
            }
            epochFileWriter.close();
            inputStream.close();
        }
        catch (Exception e) {
            System.err.println(e);
        }
    }

    public static Calendar getHeaderInfo(
            String wavFile,
            SimpleDateFormat timeFormat) {
        Calendar fileStartTime = null;
        try {
            File f = new File(wavFile);
            FileChannel file = new FileInputStream(f).getChannel();
            //For an overview on WAV file header specifications, refer to:
            //      https://ccrma.stanford.edu/courses/422/projects/WaveFormat/
            ByteBuffer buf = ByteBuffer.allocate(12); //read "RIFF" chunk
            file.read(buf);
            buf = ByteBuffer.allocate(24); //read "fmt" chunk
            file.read(buf);

            //confirm we can read "LIST" chunk
            buf = ByteBuffer.allocate(8);
            file.read(buf);
            buf.flip();
            buf.order(ByteOrder.BIG_ENDIAN);
            String header = strFromBuffer(buf,4);
            if(header.equals("LIST")) {
                //get num bytes in "LIST" chunk
                buf.order(ByteOrder.LITTLE_ENDIAN);
                int listSize = buf.getInt();
                buf = ByteBuffer.allocate(listSize);
                file.read(buf);
                buf.flip();
                buf.order(ByteOrder.BIG_ENDIAN);

                //read "LIST" chunk bytes in as a String
                header = strFromBuffer(buf,listSize);
                //for an overview on "LIST" chunk tags, please refer to:
                //      http://wiki.audacityteam.org/wiki/WAV#Metadata
                //warning: code below assumes title field comes before artist
                String deviceID = header.split("INAM|IART")[1]; //title
                String startTimeStr = header.split("IART")[1].trim(); //artist
                System.out.println("deviceID: " + deviceID);
                System.out.println("startTime: " + startTimeStr);
                fileStartTime = new GregorianCalendar();
                //fileStartTime.setTime(parseDateStr(header.split("IART")[1],timeFormat));
                fileStartTime.setTime(parseDateStr(startTimeStr,timeFormat));
            } else {
                //if no LIST chunk exists, quit as we can't get the start time
                System.out.println("No start time information!");
                System.exit(0);
            }
            file.close();
        } catch(Exception e){}
        return fileStartTime;
    }

    private static double getVectorMagnitude(double x, double y, double z) {
        return Math.sqrt(x*x + y*y + z*z)-1;
    }

    private static void abs(List<Double> vals) {
        for(int c=0; c<vals.size(); c++) {
            vals.set(c, Math.abs(vals.get(c)));
        }
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
    
    private static double mean(List<Double> vals) {
        if(vals.size()==0) {
            return Double.NaN;
        }
        return sum(vals) / (double)vals.size();
    }
    	
    private static double range(List<Double> vals) {
        if(vals.size()==0) {
            return Double.NaN;
        }
        double min = Double.MIN_VALUE;
        double max = Double.MAX_VALUE;
        for(int c=0; c<vals.size(); c++) {
            if (vals.get(c) < min) {
                min = vals.get(c);
            } else if (vals.get(c) > max) {
                max = vals.get(c);
            }
        }
        return max - min;
    }    	

    private static double std(List<Double> vals, double mean) {
        if(vals.size()==0) {
            return Double.NaN;
        }
        double var = 0; //variance
        double len = vals.size()*1.0; //length
        for(int c=0; c<vals.size(); c++) {
            var += ((vals.get(c) - mean) * (vals.get(c) - mean)) / len;
        }
        return Math.sqrt(var);
    }

    private static void writeLine(BufferedWriter fileWriter, String line) {
        try {
            fileWriter.write(line + "\n");
        } catch (Exception excep) {
            System.out.println(excep.toString());
        }
    }

    private static String strFromBuffer(ByteBuffer buf, int numChars){
        String val = "";
        for(int c=0; c<numChars; c++) {
            val += (char)buf.get() + "";
        }
        return val;
    }

    private static Date parseDateStr(String input, SimpleDateFormat timeFormat)
    {
        Date output = new Date();
        try {
            output = timeFormat.parse(input);
        } catch (Exception excep) {
            String errorMessage = "could not parse date (" + input + ")";
            errorMessage += ": " + excep.toString();
            System.out.println(errorMessage);
        }
        return output;
    }

}
