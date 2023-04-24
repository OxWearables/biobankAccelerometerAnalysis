
import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.util.Date;
import java.util.Enumeration;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.time.LocalTime;


/**
 * Calculates epoch summaries from an AX3 .CWA file.
 */
public class ActigraphReader extends DeviceReader {

    private static final int INVALID_GT3_FILE = 0;
    private static final int VALID_GT3_V1_FILE = 1;
    private static final int VALID_GT3_V2_FILE = 2;
    private static final int GT3_HEADER_SIZE = 8;

    private static Logger logger;
    static {
        String path = ActigraphReader.class.getClassLoader()
                .getResource("logging.properties")
                .getFile();
        System.setProperty("java.util.logging.config.file", path);
        logger = Logger.getLogger(ActigraphReader.class.getName());
    }


    /**
     * Reads a .gt3x file
     * For v1, the .zip archive should contain least 3 files.
     * For v2, the .zip arhive should contain only 2 files.
     * This method first verifies if it is a valid v1/v2 file,
     * it will then parse the header and begin processing the activity.bin file for v1
     * and log.bin file for v2.
     *
     * For the timestamp, since the gt3x uses .NET format even though
     * the timestamp is saved in UNIX time format but it is local time.
     * TODO: confirm the DST change
     */
    public static void readG3TXEpochs(
        String accFile,
        EpochWriter epochWriter,
        Boolean verbose) {

        ZipFile zip = null;
        // readers for the 'activity.bin' & 'info.txt' files inside the .zip
        BufferedReader infoReader = null;
        InputStream activityReader = null;

        try {
            zip = new ZipFile( new File(accFile), ZipFile.OPEN_READ);

            int gt3Version = getGT3XVersion(zip);
            if (gt3Version == INVALID_GT3_FILE) {
                System.err.println("file " + accFile + " is not a valid V1 or V2 g3tx file");
                System.exit(-2);
            }

            for (Enumeration<?> e = zip.entries(); e.hasMoreElements();) {
                ZipEntry entry = (ZipEntry) e.nextElement();
                if (entry.toString().equals("info.txt")) {
                    infoReader = new BufferedReader(new InputStreamReader(zip.getInputStream(entry)));
                } else if (entry.toString().equals("activity.bin") && gt3Version == VALID_GT3_V1_FILE) {
                    activityReader = zip.getInputStream(entry);
                } else if (entry.toString().equals("log.bin") && gt3Version == VALID_GT3_V2_FILE) {
                    activityReader = zip.getInputStream(entry);
                }
            }

            // underscored are unused for now
            double sampleFreq = -1, accelerationScale = -1, _AccelerationMin, _AccelerationMax;
            long startDate = -1, stopDate = -1, firstSampleTime=-1;
            String serialNumber = "";
            String infoTimeShift = "00:00:00"; // default to be UTC time difference

            while (infoReader.ready()) {
                String line = infoReader.readLine();
                if (line!=null){
                    String[] tokens=line.split(": ");
                    if ((tokens !=null)  && (tokens.length==2)){
                        if (tokens[0].trim().equals("Sample Rate"))
                            sampleFreq=Integer.parseInt(tokens[1].trim());
                        else if (tokens[0].trim().equals("Start Date"))
                            firstSampleTime=GT3XfromTickToMillisecond(Long.parseLong(tokens[1].trim()));
                        else if (tokens[0].trim().equals("Acceleration Scale"))
                            accelerationScale=Double.parseDouble(tokens[1].trim());
                        else if (tokens[0].trim().equals("Acceleration Min"))
                            _AccelerationMin=Double.parseDouble(tokens[1].trim());
                        else if (tokens[0].trim().equals("Acceleration Max"))
                            _AccelerationMax=Double.parseDouble(tokens[1].trim());
                        else if (tokens[0].trim().equals("Stop Date"))
                            stopDate=GT3XfromTickToMillisecond(Long.parseLong(tokens[1].trim()));
                        else if (tokens[0].trim().equals("Serial Number"))
                            serialNumber=tokens[1].trim();
                        else if (tokens[0].trim().equals("TimeZone"))
                            infoTimeShift=tokens[1].trim(); // gt3x calls time shift as time zone
                    }
                }
            }

            System.out.println("Device's initial offset: " + infoTimeShift);
            System.out.println("Start date (local UNIX): " + startDate);
            System.out.println("Stop date (local UNIX): " + stopDate);

            accelerationScale = setAccelerationScale(serialNumber);

            if ((sampleFreq==-1 || accelerationScale==-1 || firstSampleTime==-1) && gt3Version != VALID_GT3_V2_FILE) {
                System.err.println("error parsing "+accFile+", info.txt must contain 'Sample Rate', ' Start Date', and (usually) 'Acceleration Scale'.");
                System.exit(-2);
            }

            double sampleDelta = setSampleDelta(sampleFreq);

            // else leave as specified in info.txt?
            if (gt3Version == VALID_GT3_V1_FILE) readG3TXV1EpochPairs(
                    activityReader,
                    infoTimeShift,
                    sampleDelta,
                    sampleFreq,
                    accelerationScale,
                    firstSampleTime,
                    epochWriter);
            if (gt3Version == VALID_GT3_V2_FILE) readG3TXV2Epoch(
                    activityReader,
                    infoTimeShift,
                    sampleDelta,
                    sampleFreq,
                    accelerationScale,
                    epochWriter);
        } catch (IOException excep) {
            excep.printStackTrace(System.err);
            System.err.println("error reading/writing file " + accFile + ": " + excep.toString());
            System.exit(-2);
        }
        finally {
            try {
                zip.close();
                activityReader.close();
                infoReader.close();
            } catch (Exception ex) {
                /* ignore */
            }
        }
    }


    /**
     ** Method to read all the x/y/z data from a GT3X (V2) activity.bin file.
     ** File specification at: https://github.com/actigraph/NHANES-GT3X-File-Format/blob/master/fileformats/activity.bin.md
     ** Data is stored sequentially at the sample rate specified in the header (1/f = sampleDelta in milliseconds)
     ** Each pair of readings occupies an awkward 9 bytes to conserve space, so must be read 2 at a time.
     ** The readings should range from -2046 to 2046, covering -6 to 6 G's,
     ** thus the maximum accuracy is 0.003 G's. The values -2048, -2047 & 2047 should never appear in the stream.
     **/
    private static void readG3TXV2Epoch(
            InputStream activityReader,
            String infoTimeShift,
            double sampleDelta,
            double sampleFreq,
            double accelerationScale,
            EpochWriter epochWriter
    ) {

        final int PARAMETER_ID = 21;
        final int ACTIVITY_ID = 0;
        final int ACTIVITY2_ID = 26;

        int[] errCounter = new int[] { 0 }; // store val if updated in other
        // method (pass by reference using array?)

        // Read 2 XYZ samples at a time, each sample consists of 36 bits ... 2 full samples will be 9 bytes
        int checkSum = 0, type=0;
        int i = 0;
        long date = 0;
        int datum;
        int separator = 0;
        int size = 0;
        int initIndex = 0; // starting index of a parket
        boolean isHeader = true;
        int packetCount = 0;

        // 1. process header
        // 2. process payload based on type for each packet
        // 3. validate checksum for each packet
        try {
            while ((datum=activityReader.read())!=-1){
                // 1. Process header
                byte current = (byte)datum;
                if (isHeader) {
                    switch (i-initIndex) {
                        case 0:
                            separator = current;
                            break;
                        case 1:
                            type = current;
                            break;
                        case 2:
                            // not sure why this is needed but without casting, this might
                            // result in leading ones during type conversion
                            date = (long)(current & 0xFF);
                            break;
                        case 3:
                            date = (long)(((current & 0xFF) << 8) ^ date);
                            break;
                        case 4:
                            date = (long)(((current & 0xFF) << 16) ^ date);
                            break;
                        case 5:
                            date = (long)(((current & 0xFF) << 24) ^ date);
                            break;
                        case 6:
                            size = (int)(current & 0xFF);
                            break;
                        case 7:
                            size = (int)(((current & 0xFF) << 8) ^ size);
                    }

                    if (i == initIndex+GT3_HEADER_SIZE-1) {
                        isHeader = false;
                        logger.log(Level.FINER, "\nHeader info" +
                                "\ntype: "+ type +
                                String.format("\nDate 0x%08X: ", date) +
                                "\nsize: "+ size +
                                "\nStarting index: "+ initIndex);
                    }

                } else if (isPayload(i, size, initIndex)) {
                    // process payload depending on the type of record
                    // There exist various packet types. Currently, we are
                    // only processing packets of type ACTIVITY
                    // https://github.com/actigraph/GT3X-File-Format
                    checkSum ^= (byte)current;

                    if (type == PARAMETER_ID) {
                        logger.log(Level.INFO, "Processing parameter packet...");
                        // process parameters Keyvale pair. Each pair is of 8 bytes.
                        byte [] keyPair = new byte[8];
                        byte mydatum;
                        keyPair[0] = current;

                        int k = 1;
                        while (k < 8) {
                            mydatum= (byte) activityReader.read();
                            keyPair[k] = (byte) mydatum;
                            checkSum ^= (byte) mydatum;
                            k++;
                        }

                        // set acceleration scale if present
                        if (isAccelScale(keyPair)) {
                            int keyval = keyPair[4];
                            keyval = (int)(((keyPair[5] & 0xFF) << 8) ^ keyval);
                            keyval = (int)(((keyPair[6] & 0xFF) << 16) ^ keyval);
                            keyval = (int)(((keyPair[7] & 0xFF) << 24) ^ keyval);
                            accelerationScale = decodePara(keyval);
                            logger.log(Level.INFO, "accelerationScale changed to "+accelerationScale);
                        }

                        i += 7;
                    } else if (type == ACTIVITY_ID && size > 1) {
                        // when Size = 1, it is a USB connection event thus ignore.
                        int [] res = processActivity(
                                infoTimeShift,
                                sampleFreq,
                                date,
                                current,
                                i,
                                size,
                                checkSum,
                                initIndex,
                                accelerationScale,
                                activityReader,
                                epochWriter);
                        i = res[0];
                        checkSum = res[1];
                    } else if (type == ACTIVITY2_ID && size > 1) {
                        // when Size = 1, it is a USB connection event thus ignore.
                        int [] res = processActivity2(
                                infoTimeShift,
                                sampleFreq,
                                date,
                                current,
                                i,
                                size,
                                checkSum,
                                initIndex,
                                accelerationScale,
                                activityReader,
                                epochWriter);
                        i = res[0];
                        checkSum = res[1];
                    }
                } else {
                    checkChecksum(i, separator, type, size, date, checkSum, current);
                    checkSum = 0;
                    date = 0;
                    size = 0;
                    type = 0;
                    separator = 0;
                    // allow reading header after checksum is done checking
                    isHeader = true;
                    initIndex = i+1;
                    packetCount++;

                    if (packetCount % 10000 == 0) {
                        logger.log(Level.INFO, "Done processing "+packetCount+" packets.");
                    }
                }

                i++;
            }
        }
        catch (IOException ex) {
            logger.log(Level.INFO, "End of .g3tx file reached");
        }
    }

    
    /**
     ** Method to read all the x/y/z data from a GT3X (V1) activity.bin file.
     ** File specification at: https://github.com/actigraph/NHANES-GT3X-File-Format/blob/master/fileformats/activity.bin.md
     ** Data is stored sequentially at the sample rate specified in the header (1/f = sampleDelta in milliseconds)
     ** Each pair of readings occupies an awkward 9 bytes to conserve space, so must be read 2 at a time.
     ** The readings should range from -2046 to 2046, covering -6 to 6 G's,
     ** thus the maximum accuracy is 0.003 G's. The values -2048, -2047 & 2047 should never appear in the stream.
     **/
    private static void readG3TXV1EpochPairs(
            InputStream activityReader,
            String infoTimeShift,
            double sampleDelta,
            double sampleFreq,
            double accelerationScale,
            long firstSampleTime, // in milliseconds
            EpochWriter epochWriter
            ) {

        int[] errCounter = new int[] { 0 }; // store val if updated in other
                                            // method (pass by reference using array?)
        int samples = 0; // num samples collected so far

        // Read 2 XYZ samples at a time, each sample consists of 36 bits ... 2 full samples will be 9 bytes
        byte[] bytes=new byte[9];
        int i=0;
        int twoSampleCounter = 0;
        int datum;
        int totalBytes = 0;
        double[] twoSamples = null;

        try {
            while (( datum=activityReader.read())!=-1){
                bytes[i]=(byte)datum;
                totalBytes++;


                if (false && totalBytes%10000==0)
                    System.out.println("Converting sample.... "+(totalBytes/1000)+"K");

                // if we have enough bytes to read two 36 bit data samples
                if (++i==9){
                    twoSamples = readAccelPair(bytes, accelerationScale);
                    twoSampleCounter = 2;
                }

                // read the two samples from the sample counter
                while (twoSampleCounter>0) {
                    twoSampleCounter--;
                    i=0;

                    long time = Math.round((1000d*samples)/sampleFreq) + firstSampleTime;
                    double x = twoSamples[3-twoSampleCounter*3];
                    double y = twoSamples[4-twoSampleCounter*3];
                    double z = twoSamples[5-twoSampleCounter*3];
                    double temp = 1.0d; // don't know temp yet
                    time = getTrueUnixTime(time, infoTimeShift);
                    epochWriter.newValues(time, x, y, z, temp, errCounter);

                    samples += 1;
                }
            }
        }
        catch (IOException ex) {
            System.out.println("End of .g3tx file reached");
        }
    }


    private static int [] processActivity(
            String infoTimeShift,
            double sampleFreq,
            long firstSampleTime,
            byte current,
            int i,
            int size,
            int checkSum,
            int initIndex,
            double accelerationScale,
            InputStream activityReader,
            EpochWriter epochWriter) {
        int[] errCounter = new int[] { 0 }; // store val if updated in other

        double [] sample = new double[3];
        int offset = 0;
        int shifter;
        short axis_val;
        int samples = 0;
        try {
            while (isPayload(i, size, initIndex)) {
                for (int axis = 0; axis < 3; axis++) {
                    if (0 == (offset & 0x07)) {
                        if (i != initIndex + GT3_HEADER_SIZE) {
                            current = (byte) activityReader.read();
                            checkSum ^= (byte) current;
                        }
                        i++;

                        shifter = ((current & 0xFF) << 4);

                        current = (byte) activityReader.read();
                        checkSum ^= (byte) current;
                        i++;

                        shifter |= ((current & 0xF0) >>> 4);
                        offset += 12;
                    } else {
                        shifter = ((current & 0x0F) << 8);

                        current = (byte) activityReader.read();
                        checkSum ^= (byte) current;
                        i++;
                        shifter |= (current & 0xFF);
                        offset += 12;
                    }
                    if (shifter > 2047)
                        shifter += 61440;

                    axis_val = (short) shifter;
                    sample[axis] = axis_val / accelerationScale;
                    sample[axis] = (double) Math.round(sample[axis] * 1000d) / 1000d; // round to 3rd decimal
                }
                logger.log(Level.FINER, "i: " + i);
                logger.log(Level.FINER, "x y z: " + sample[1] + " " + sample[0] + " " + sample[2]);

                double temp = 1.0d; // don't know temp yet
                samples += 1;
                long myTime = Math.round((1000d*samples)/sampleFreq) + firstSampleTime*1000; // in Miliseconds
                myTime = getTrueUnixTime(myTime, infoTimeShift);
                epochWriter.newValues(myTime, sample[1], sample[0], sample[2], temp, errCounter);
            }
        } catch (IOException ex) {
            ex.printStackTrace(System.err);
            System.err.println("error when reading activity at byte " + i + ": " + ex.toString());
            System.exit(-2);
        }

        return new int[] {i, checkSum};
    }


    public static long getTrueUnixTime(long myTime, String infoTimeShift) {
        int shiftSign = 1;
        if (infoTimeShift.charAt(0) == '-') {
            shiftSign = -1;
            infoTimeShift = infoTimeShift.substring(1);
        }

        LocalTime timeShift = LocalTime.parse(infoTimeShift);
        long timeShiftMilli = 1000 * (shiftSign * timeShift.getHour() * 60 * 60 +
                timeShift.getMinute() * 60); // time shfit w.r.t. UTC
        return myTime - timeShiftMilli;
    }


    private static int [] processActivity2(
            String infoTimeShift,
            double sampleFreq,
            long firstSampleTime,
            byte current,
            int i,
            int size,
            int checkSum,
            int initIndex,
            double accelerationScale,
            InputStream activityReader,
            EpochWriter epochWriter) {
        int[] errCounter = new int[] { 0 }; // store val if updated in other

        double [] sample = new double[3];
        int shifter;
        short axis_val;
        int samples = 0;
        try {
            while (isPayload(i, size, initIndex)) {
                for (int axis = 0; axis < 3; axis++) {
                    if (i != initIndex + GT3_HEADER_SIZE) {
                        current = (byte) activityReader.read();
                        checkSum ^= (byte) current;
                    }

                    shifter = current & 0xff;
                    current = (byte) activityReader.read();
                    checkSum ^= (byte) current;
                    shifter |= ((current & 0xff) << 8);
                    i += 2;

                    axis_val = (short) shifter;

                    sample[axis] = axis_val / accelerationScale;
                    sample[axis] = (double) Math.round(sample[axis] * 1000d) / 1000d; // round to 3rd decimal
                }

                double temp = 1.0d; // don't know temp yet
                samples += 1;

                long myTime = Math.round((1000d*samples)/sampleFreq) + firstSampleTime*1000; // in Miliseconds
                myTime = getTrueUnixTime(myTime, infoTimeShift);

                logger.log(Level.FINER, "i: " + i + "\nx y z: " + sample[0] + " " + sample[1] + " " + sample[2] +
                        "\nTime:" + myTime);
                epochWriter.newValues(myTime,
                                      sample[0], sample[1], sample[2], temp, errCounter);
            }
        } catch (IOException ex) {
            ex.printStackTrace(System.err);
            System.err.println("error when reading activity at byte " + i + ": " + ex.toString());
            System.exit(-2);
        }

        return new int[] {i, checkSum};
    }


    private static double[] readAccelPair(byte[] bytes, double accelerationScale) {

        int datum = 0;
        datum=(bytes[0]&0xff);datum=datum<<4;datum|=(bytes[1]&0xff)>>>4;
        short y1=(short)datum;
        if (y1>2047)
            y1+=61440;

        datum=bytes[1]&0x0F;datum=datum<<8;datum|=(bytes[2]&0xff);
        short x1=(short)datum;
        if (x1>2047)
            x1+=61440;

        datum=bytes[3]&0xff;datum=datum<<4;datum|=(bytes[4]&0xff)>>>4;
        short z1=(short)datum;
        if (z1>2047)
            z1+=61440;

        datum=bytes[4]&0x0F;datum=datum<<8;datum|=(bytes[5]&0xff);
        short y2=(short) datum;
        if (y2>2047)
            y2+=61440;

        datum=(bytes[6]&0xff);datum=datum<<4;datum|=(bytes[7]&0xff)>>>4;
        short x2=(short)datum;
        if (x2>2047)
            x2+=61440;

        datum=bytes[7]&0x0F;datum=datum<<8;datum|=(bytes[8]&0xff);
        short z2=(short)datum;
        if (z2>2047)
            z2+=61440;

        // convert to 'g'
        double gx1=x1/accelerationScale;
        double gy1=y1/accelerationScale;
        double gz1=z1/accelerationScale;

        double gx2=x2/accelerationScale;
        double gy2=y2/accelerationScale;
        double gz2=z2/accelerationScale;

        return new double[] {gx1, gy1, gz1, gx2, gy2, gz2};
    }


    /**
     * check checksum with payload and header info
     */
    private static void checkChecksum(
            int i,
            int separator,
            int type,
            int size,
            long date,
            int checkSum,
            int target_value) {

        checkSum ^= (byte)separator;
        checkSum ^= (byte)type;
        checkSum ^= (byte)(size & 0xFF);
        checkSum ^= (byte)((size >> 8) & 0xFF);
        checkSum ^= (byte)(date & 0xFF);
        checkSum ^= (byte)((date >> 8) & 0xFF);
        checkSum ^= (byte)((date >> 16) & 0xFF);
        checkSum ^= (byte)((date >> 24) & 0xFF);

        // to convert to one's complement as the checksum is one's complement
        checkSum = (byte)~checkSum;
        if (checkSum != target_value) {
            logger.log(Level.SEVERE, "Packet parsing failed at byte "+ i + "\nChecksum does not match!"
                    + String.format("\nExpected 0x%08X", target_value) +String.format("\nObtained 0x%08X", checkSum));
            System.exit(-1);
        } else {
            logger.log(Level.FINER, "Verification succeeds");
        }
    }


    private static double setAccelerationScale(String serialNumber) {
        double ACCELERATION_SCALE_FACTOR_NEO_CLE = 341.0; // == 2046 (range of data) / 6 (range of G's)
        double ACCELERATION_SCALE_FACTOR_MOS = 256.0; // == 2048/8?
        double accelerationScale = -1;

        if((serialNumber.startsWith("NEO") || (serialNumber.startsWith("CLE")))) {
            accelerationScale = ACCELERATION_SCALE_FACTOR_NEO_CLE;
        } else if(serialNumber.startsWith("MOS")){
            accelerationScale = ACCELERATION_SCALE_FACTOR_MOS;
        }
        return accelerationScale;
    }


    private static double setSampleDelta(double sampleFreq) {
        System.out.println("sampleFreq:" + sampleFreq);
        double sampleDelta_old = Math.round(1000.0/sampleFreq * 100d) / 100d;  // round the delta to its fourth decimal
        System.out.println("sampleDelta before rounding:" + sampleDelta_old);
        double sampleDelta = 1000.0/sampleFreq;  // don't round the delta to its fourth decimal
        System.out.println("sampleDelta after rounding:" + sampleDelta);
        //  sampleDelta = Math.round(1000.0/sampleFreq * 100d) / 100d;  // round the delta to its fourth decimal

        return sampleDelta;
    }


    /**
     ** Payload should between the initIndex and InitIndex+size. Upper bound is
     *  exclusive.
     **
     */
    private static boolean isPayload(int i, int size, int initIndex) {
        if (i >= initIndex+GT3_HEADER_SIZE && i < (initIndex+GT3_HEADER_SIZE+size)) return true;
        else return false;
    }

    /**
     * This was translated into Java from
     * https://github.com/actigraph/GT3X-File-Format/blob/master/LogRecords/Parameters.md
     */
    public static double decodePara(int value) {
        final double FLOAT_MAXIMUM = 8388608.0;                  /* 2^23  */
        final int ENCODED_MINIMUM = 0x00800000;
        final int ENCODED_MAXIMUM = 0x007FFFFF;
        final int SIGNIFICAND_MASK = 0x00FFFFFF;
        final int EXPONENT_MASK = 0xFF000000;
        final int EXPONENT_OFFSET = 24;

        double significand;
        int exponent;
        int i32;

        /* handle numbers that are too big */
        if (ENCODED_MAXIMUM == value)
            return Integer.MAX_VALUE;
        else if (ENCODED_MAXIMUM == value)
            return -Integer.MAX_VALUE;

        /* extract the exponent */
        i32 = (int) ((value & EXPONENT_MASK) >>> EXPONENT_OFFSET);
        if (0 != (i32 & 0x80))
            i32 = (int)((int)i32 | 0xFFFFFF00);
        exponent = (int)i32;

        /* extract the significand */
        i32 = (int)(value & SIGNIFICAND_MASK);
        if (0 != (i32 & ENCODED_MINIMUM))
            i32 = (int)((int)i32 | 0xFF000000);

        significand = (double) i32 / FLOAT_MAXIMUM;

        /* calculate the floating point value */
        return significand * Math.pow(2.0, exponent);
    }


    private static boolean isAccelScale(byte[] keyPairs) {
        int addressSpace = keyPairs[0];
        int identifier = keyPairs[2];
        if (addressSpace == 0 && identifier == 55) return true;
        else return false;
    }


    /**
     ** Helper method that converts .NET ticks that Actigraph GT3X uses to millisecond (local)
     ** method from: https://github.com/SPADES-PUBLIC/mHealth-GT3X-converter-public/blob/master/src/com/qmedic/data/converter/gt3x/GT3XUtils.java
     *
     * Unit: .NET has a unit of 100 naooseconds
     * https://docs.microsoft.com/en-us/dotnet/api/system.datetime.ticks?view=netcore-3.1
     **/
    public static long GT3XfromTickToMillisecond(final long ticks)
    {
        Date date = new Date((ticks - 621355968000000000L) / 10000);
        return date.getTime();
    }


    /*
     * This method checks which GT3 version the zipfile contains.
     * Return 1 for v1, 2 for v2, 0 for invalid GT3 file
     */
    private static int getGT3XVersion(final ZipFile zip) throws IOException {

        // Check if the file contains the necessary Actigraph files
        boolean hasActivityData = false;
        boolean hasLuxData = false;
        boolean hasInfoData = false;
        boolean hasLogData = false;
        for (Enumeration<?> e = zip.entries(); e.hasMoreElements();) {
            ZipEntry entry = (ZipEntry) e.nextElement();
            if (entry.toString().equals("activity.bin"))
                hasActivityData = true;
            if (entry.toString().equals("lux.bin"))
                hasLuxData = true;
            if (entry.toString().equals("info.txt"))
                hasInfoData = true;
            if (entry.toString().equals("log.bin"))
                hasLogData = true;
        }

        if (hasActivityData && hasLuxData && hasInfoData) {
            return VALID_GT3_V1_FILE;
        } else if (hasInfoData && hasLogData) {
            return VALID_GT3_V2_FILE;
        }

        return INVALID_GT3_FILE;
    }

    
}