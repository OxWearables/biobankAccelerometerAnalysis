
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.time.format.DateTimeFormatter;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.zone.ZoneRules;
import java.time.ZoneOffset;
import java.util.concurrent.TimeUnit;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Common methods to calculate epoch summaries across all device types.
 */
public class DeviceReader {

    protected static ZonedDateTime sessionStart = null;
    protected static boolean sessionStartDST;

    protected static ZoneId zoneId;
    protected static ZoneRules rules;

    protected static int timeShift = 0;


    public static EpochWriter setupEpochWriter(
        String outputFile,
        boolean useFilter,
        boolean rawOutput,
        String rawFile,
        boolean npyOutput,
        String npyFile,
        boolean getFeatures,
        DateTimeFormatter timeFormat,
        String timeZone,
        int epochPeriod,
        int sampleRate,
        String resampleMethod,
        int range,
        double[] xyzIntercept,
        double[] xyzSlope,
        double[] xyzSlopeT,
        boolean getStationaryBouts,
        double stationaryStd,
        long startTime,
        long endTime,
        boolean verbose
        ){

        // file read/write objects
        EpochWriter epochWriter = null;
        BufferedWriter epochFileWriter = null;
        BufferedWriter rawWriter = null; // raw and npy are null if not used
        NpyWriter npyWriter = null;
        try{
            if (outputFile.endsWith(".gz")) {
            GZIPOutputStream zip = new GZIPOutputStream(new FileOutputStream(new File(outputFile)));
            epochFileWriter = new BufferedWriter(new OutputStreamWriter(zip, "UTF-8"));
        } else {
            epochFileWriter = new BufferedWriter(new FileWriter(outputFile));
        }
        LowpassFilter filter = null;
        if (useFilter) {
            filter = new LowpassFilter(20, sampleRate, verbose);
        }
        if (rawOutput) {
            if (rawFile.trim().length() == 0) {
                rawFile = (outputFile.toLowerCase().endsWith(".csv.gz") // generate raw output filename
                    ? outputFile.substring(0, outputFile.length() - ".csv.gz".length()) : outputFile) + "_raw.csv.gz";
            }
            GZIPOutputStream zipRaw = new GZIPOutputStream(
                                new FileOutputStream(new File(rawFile)));
            rawWriter = new BufferedWriter(new OutputStreamWriter(zipRaw, "UTF-8"));
        }
        if (npyOutput) {
            if (npyFile.trim().length() == 0) {
                npyFile = (outputFile.toLowerCase().endsWith(".csv.gz") // generate npy output filename
                    ? outputFile.substring(0, outputFile.length() - ".csv.gz".length()) : outputFile) + "_raw.npy";
            }
            npyWriter = new NpyWriter(npyFile);
        }
        epochWriter = new EpochWriter(
                  epochFileWriter,
                  rawWriter,
                  npyWriter,
                  timeFormat,
                  timeZone,
                  epochPeriod,
                  sampleRate,
                  resampleMethod,
                  range,
                  xyzIntercept,
                  xyzSlope,
                  xyzSlopeT,
                  getStationaryBouts,
                  stationaryStd,
                  filter,
                  startTime,
                  endTime,
                  getFeatures);
        } catch (IOException excep) {
            excep.printStackTrace(System.err);
            System.err.println("error closing file writer: " + excep.toString());
            System.exit(-2);
        }

        return epochWriter;
    }


    protected static long getUncompressedSizeofGzipFile(String gzipFile){
        //credit to https://stackoverflow.com/questions/7317243/gets-the-uncompressed-size-of-this-gzipinputstream
        long val = 500000;
        try{
            RandomAccessFile raf = new RandomAccessFile(gzipFile, "r");
            raf.seek(raf.length() - Integer.BYTES);
            int n = raf.readInt();
            val = Integer.toUnsignedLong(Integer.reverseBytes(n));
        } catch(IOException excep){
            excep.printStackTrace(System.err);
            System.err.println("error reading gz size of file " + gzipFile + ": " + excep.toString());
        }
        return val;
    }


    // Convert LocalDateTime to epoch milliseconds (from 1970 epoch)
    protected static long getEpochMillis(LocalDateTime date) {
        return date.toInstant(ZoneOffset.UTC).toEpochMilli();
    }


    protected static long secs2Nanos(double num) {
        return (long) (TimeUnit.SECONDS.toNanos(1) * num);
    }


    // credit for next 2 methods goes to:
    // http://stackoverflow.com/questions/9883472/is-it-possiable-to-have-an-unsigned-bytebuffer-in-java
    protected static long getUnsignedInt(ByteBuffer bb, int position) {
        return ((long) bb.getInt(position) & 0xffffffffL);
    }

    protected static int getUnsignedShort(ByteBuffer bb, int position) {
        return (bb.getShort(position) & 0xffff);
    }


    protected static int getSignedIntFromHex(String dataBlock, int startPos, int length) {
        // input hex base is 16
        int rawVal = Integer.parseInt(dataBlock.substring(startPos, startPos + length), 16);
        int unsignedLimit = 4096; // 2^[length*4] #i.e. 3 hexBytes (12 bits)
                                    // limit = 4096
        int signedLimit = 2048; // 2^[length*(4-1)] #i.e. 3 hexBytes - 1 bit (11
                                // bits) limit = 2048
        if (rawVal > signedLimit) {
            rawVal = rawVal - unsignedLimit;
        }
        return rawVal;
    }


    protected static void setTimeSettings(String timeZone, int timeShift) {
        DeviceReader.zoneId = ZoneId.of(timeZone);
        DeviceReader.rules = zoneId.getRules();
        DeviceReader.timeShift = timeShift;
    }


    protected static void setSessionStart(LocalDateTime ldt) {
        DeviceReader.sessionStart = ldt.atZone(zoneId);
        DeviceReader.sessionStartDST = rules.isDaylightSavings(sessionStart.toInstant());
    }


    protected static ZonedDateTime zonedWithDSTCorrection(LocalDateTime ldt) {
        ZonedDateTime zdt = ldt.atZone(zoneId);
        boolean isDST = rules.isDaylightSavings(zdt.toInstant());
        if (isDST != sessionStartDST) {  // DST crossover happened
            zdt = isDST ? ldt.plusHours(1).atZone(zoneId) : zdt.plusHours(-1);
        }
        return zdt;
    }


}
