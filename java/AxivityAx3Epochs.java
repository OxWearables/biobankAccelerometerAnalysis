
//BSD 2-Clause (c) 2014: A.Doherty (Oxford), D.Jackson, N.Hammerla (Newcastle)
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import java.io.OutputStreamWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.math.RoundingMode;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.Enumeration;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.TimeZone;
import java.time.Duration;
import java.time.format.DateTimeFormatter;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.text.DecimalFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;


/**
 * Calculates epoch summaries from an AX3 .CWA file. Class/application can be
 * called from the command line as follows: java AxivityAx3Epochs inputFile.CWA
 */
public class AxivityAx3Epochs {

	private static final DecimalFormat DF6 = new DecimalFormat("0.000000");
	private static final DecimalFormat DF3 = new DecimalFormat("0.000");
	private static final DecimalFormat DF2 = new DecimalFormat("0.00");

	private static Boolean verbose = true;
	private static EpochWriter epochWriter;

	/*
	 * Parse command line args, then call method to identify and write epochs.
	 *
	 * @param args
	 *            An argument string passed in by ActivitySummary.py. Contains
	 *            "param:value" pairs.
	 */
	public static void main(String[] args) {


		// variables to store default parameter options
		String[] functionParameters = new String[0];

		String accFile = ""; // file to process
		String outputFile = ""; // file name for epoch file
		String rawFile = ""; // file name for epoch file
		String npyFile = ""; // file name for epoch file
		Boolean rawOutput = false; // whether to output raw data
		Boolean fftOutput = false; // whether to output fft data
		Boolean npyOutput = false; // whether to output npy data

		DF6.setRoundingMode(RoundingMode.CEILING);
		DF2.setRoundingMode(RoundingMode.CEILING);
    	DF3.setRoundingMode(RoundingMode.HALF_UP); // To match their implementation

    	// epochWriter parameters
    	int epochPeriod = 5;
    	DateTimeFormatter timeFormat = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS");
    	DateTimeFormatter csvTimeFormat = DateTimeFormatter.ofPattern("M/d/yyyy,HH:mm:ss.SSSS");
    	boolean getEpochCovariance = false;
    	boolean getAxisMeans = false;
    	boolean getStationaryBouts = false;
    	double stationaryStd = 0.013;
    	double[] swIntercept = new double[] { 0.0, 0.0, 0.0 };
    	double[] swSlope = new double[] { 1.0, 1.0, 1.0 };
    	double[] tempCoef = new double[] { 0.0, 0.0, 0.0 };
    	double meanTemp = 0.0;
    	int range = 8;
    	int sampleRate = 100;
    	boolean useFilter = true;
    	long startTime = -1; // milliseconds since epoch
    	long endTime = -1;
    	int timeZoneOffset = 0;
    	boolean getSanDiegoFeatures = false;
    	boolean getMADFeatures = false;
    	boolean getUnileverFeatures = false;
    	boolean get3DFourier = true;
    	boolean getEachAxis = true;
    	boolean useAbs = false;
    	int numEachAxis = 15; // number of fft bins to print

    	// Must supply additional information when loading from a .csv file
    	LocalDateTime csvStartTime = null; // start date of first sample
    	double csvSampleRate = -1; // must specify sample rate if time column and date format not given
    	int csvStartRow = 0;
    	// [0, 1, 2] = assume X is 1st column, Y is 2nd, Z is 3rd, and no time column.
    	// if list is longer than 3 elements the extra elements are the columns that make up a time value
    	// e.g. [0,1,2,3,6] means use 1st three columns for X/Y/Z, and concatenate cols 3 & 6 to parse time
    	// (useful when time and date are in separate columns)
    	List<Integer> csvXYZTCols = Arrays.asList( 0,1,2 );

    	// file read/write objects
    	BufferedWriter epochFileWriter = null;
    	BufferedWriter rawWriter = null; // raw and fft are null if not used
    	BufferedWriter fftWriter = null; // else will be similar to epochFile name
    	NpyWriter npyWriter = null;


		if (args.length < 1) {
			String invalidInputMsg = "Invalid input, ";
			invalidInputMsg += "please enter at least 1 parameter, e.g.\n";
			invalidInputMsg += "java AxivityAx3Epochs inputFile.CWA";
			System.out.println(invalidInputMsg);
			System.exit(-1);
		}

		if (args.length == 1) {
			// single parameter needs to be accFile
			accFile = args[0];
			outputFile = accFile.split("\\.")[0] + "Epoch.csv";
		} else {
			// load accFile, and also copy functionParameters (args[1:])
			accFile = args[0];
			outputFile = accFile.split("\\.")[0] + "Epoch.csv";
			functionParameters = Arrays.copyOfRange(args, 1, args.length);
			SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-M-d'T'H:m");
			dateFormat.setTimeZone(TimeZone.getTimeZone("UTC"));

			// update default values by looping through available user
			// parameters
			for (String param : functionParameters) {
				// individual_Parameters will look like "epoch_period:60"
				String funcName = param.split(":")[0];
				String funcParam = param.substring(param.indexOf(":") + 1);
				if (funcName.equals("outputFile")) {
					outputFile = funcParam;
				} else if (funcName.equals("timeZoneOffset")) {
					timeZoneOffset = Integer.parseInt(funcParam);
				} else if (funcName.equals("verbose")) {
					verbose = Boolean.parseBoolean(funcParam.toLowerCase());
				} else if (funcName.equals("epochPeriod")) {
					epochPeriod = Integer.parseInt(funcParam);
				// Make sure time f
				} else if (funcName.equals("timeFormat")) {
					timeFormat = DateTimeFormatter.ofPattern(funcParam);
				} else if (funcName.equals("filter")) {
					useFilter = Boolean.parseBoolean(funcParam.toLowerCase());
				} else if (funcName.equals("getEpochCovariance")) {
					getEpochCovariance = Boolean.parseBoolean(funcParam.toLowerCase());
				} else if (funcName.equals("getAxisMeans")) {
					getAxisMeans = Boolean.parseBoolean(funcParam.toLowerCase());
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
				} else if (funcName.equals("sampleRate")) {
					sampleRate= Integer.parseInt(funcParam);
				} else if (funcName.equals("range")) {
					range = Integer.parseInt(funcParam);
				} else if (funcName.equals("rawOutput")) {
					rawOutput = Boolean.parseBoolean(funcParam.toLowerCase());
				} else if (funcName.equals("rawFile")) {
					rawFile = funcParam;
				} else if (funcName.equals("fftOutput")) {
					fftOutput = Boolean.parseBoolean(funcParam.toLowerCase());
				} else if (funcName.equals("npyOutput")) {
                    npyOutput = Boolean.parseBoolean(funcParam.toLowerCase());
                } else if (funcName.equals("npyFile")) {
					npyFile = funcParam;
				} else if (funcName.equals("useAbs")) {
					useAbs = Boolean.parseBoolean(funcParam.toLowerCase());
				} else if (funcName.equals("startTime")) {
					try {
						startTime = dateFormat.parse(funcParam).getTime();
					} catch (ParseException ex) {
						System.err.println("error parsing startTime:'"+funcParam+"', must be in format: 1996-7-30T13:59");
						System.exit(-2);
					}
				} else if (funcName.equals("endTime")) {
					try {
						endTime = dateFormat.parse(funcParam).getTime();
					} catch (ParseException ex) {
						System.err.println("error parsing endTime:'"+funcParam+"', must be in format: 1996-7-30T13:59");
						System.exit(-2);
					}
				} else if (funcName.equals("csvXYZTCols")) {
					String[] XYZT = funcParam.split(",");
					if (XYZT.length < 3) {
						System.err.println("error parsing csvXYZTCols: "+XYZT.toString()+"\n must be 3 or 4 integers");
						System.exit(-2);
					}
					csvXYZTCols = new LinkedList<Integer>();
					System.out.println(csvXYZTCols.toString());
					for( int i = 0; i<XYZT.length; i++ ) {
						System.out.println(XYZT[i] + " - " + Integer.parseInt(XYZT[i]));
						csvXYZTCols.add(Integer.parseInt(XYZT[i]));
					}

				} else if (funcName.equals("csvStartRow")) {
					csvStartRow = Integer.parseInt(funcParam);
				} else if (funcName.equals("getSanDiegoFeatures")) {
					getSanDiegoFeatures = Boolean.parseBoolean(funcParam.toLowerCase());
				} else if (funcName.equals("getMADFeatures")) {
					getMADFeatures = Boolean.parseBoolean(funcParam.toLowerCase());
				} else if (funcName.equals("getUnileverFeatures")) {
					getUnileverFeatures = Boolean.parseBoolean(funcParam.toLowerCase());
				} else if (funcName.equals("get3DFourier")) {
					get3DFourier = Boolean.parseBoolean(funcParam.toLowerCase());
				} else if (funcName.equals("getEachAxis")) {
					getEachAxis = Boolean.parseBoolean(funcParam.toLowerCase());
				} else if (funcName.equals("numEachAxis")) {
					numEachAxis = Integer.parseInt(funcParam);
				} else if (funcName.equals("csvStartTime")) {
					try {
						csvStartTime = epochMillisToLocalDateTime( dateFormat.parse(funcParam).getTime() );
					} catch (ParseException ex) {
						System.err.println("error parsing startTime, must be in format: 1996-7-30T13:01");
						System.exit(-2);
					}
				} else if (funcName.equals("csvTimeFormat")) {
					csvTimeFormat = DateTimeFormatter.ofPattern(funcParam);
				} else if (funcName.equals("csvSampleRate")) {
					csvSampleRate = Double.parseDouble(funcParam);
				} else {
					System.err.println("unknown parameter " + funcName + ":" + funcParam);
				}

			}
		}


		try {
			System.out.println("Intermediate file: " + outputFile);
            if (outputFile.endsWith(".gz")) {
                GZIPOutputStream zip = new GZIPOutputStream(new FileOutputStream(new File(outputFile)));
                epochFileWriter = new BufferedWriter(new OutputStreamWriter(zip, "UTF-8"));
            } else {
                epochFileWriter = new BufferedWriter(new FileWriter(outputFile));
            }
            LowpassFilter filter = null;
			if (useFilter) {
				filter = new LowpassFilter(20 /* = lowPassCutFrequency*/, sampleRate /* = sampleRate*/);
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
			if (fftOutput && !getSanDiegoFeatures) {
				String fftFile = (outputFile.toLowerCase().endsWith(".csv") // generate raw output filename
						? outputFile.substring(0, outputFile.length() - ".csv".length()) : outputFile) + "_fft.csv";
				fftWriter = new BufferedWriter(new FileWriter(fftFile));
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
				      fftWriter,
				      npyWriter,
				      timeFormat,
				      epochPeriod,
				      sampleRate,
				      range,
				      swIntercept,
				      swSlope,
				      tempCoef,
				      meanTemp,
				      getStationaryBouts,
				      stationaryStd,
				      filter,
				      getEpochCovariance,
				      getAxisMeans,
				      startTime,
				      endTime,
				      getSanDiegoFeatures,
				      getMADFeatures,
				      getUnileverFeatures,
				      get3DFourier,
				      getEachAxis,
				      numEachAxis,
				      useAbs
				      );

			// process file if input parameters are all ok
			if (accFile.toLowerCase().endsWith(".cwa")) {
				readCwaEpochs(accFile, timeZoneOffset);
			} else if (accFile.toLowerCase().endsWith(".cwa.gz")) {
                readCwaGzEpochs(accFile, timeZoneOffset);
            } else if (accFile.toLowerCase().endsWith(".bin")) {
				readGeneaEpochs(accFile);
			} else if (accFile.toLowerCase().endsWith(".gt3x")) {
				readG3TXEpochs(accFile);
			} else if (accFile.toLowerCase().endsWith(".csv")) {
				if ((csvStartTime==null || csvSampleRate<=0) && (csvTimeFormat==null || csvXYZTCols.size()<=3)) {
					System.err.println("Must specify either 'csvStartTime' (YYYY-MM-DDThh:mm format) and 'csvSampleRate', or:\n"
									 + "'csvXYZYCols' (at least 4 values) and 'csvTimeFormat'\n"
									 + "options for .csv type file: " + accFile);
					System.exit(-1);
				}
				readCSVEpochs(accFile, csvStartTime, csvSampleRate, csvTimeFormat, csvStartRow, csvXYZTCols);
			} else {
				System.err.println("Unrecognised file format for: " + accFile);
				System.exit(-1);
			}
		} catch (IOException excep) {
			excep.printStackTrace(System.err);
			System.err.println("error reading/writing file " + outputFile + ": " + excep.toString());
			System.exit(-2);
		} finally {
			try {
				epochFileWriter.close();
				if (rawOutput) rawWriter.close();
				if (fftOutput) fftWriter.close();
				if (npyOutput) npyWriter.close();
			} catch (Exception ex) {
				/* ignore */
			}
		}

		// if no errors then return success code
		System.exit(0);
	}


	private static void readCSVEpochs(
			String accFile,
			LocalDateTime startTime,
			double csvSampleRate,
			DateTimeFormatter csvTimeFormat,
			int csvStartRow,
			List<Integer> csvXYZTCols) {

		try {
			BufferedReader accStream =  new BufferedReader(new FileReader(accFile));
			String line = "";
			int lineNumber = 0;
			String csvSplitBy = ",";
			int numColsRequired = Collections.max(csvXYZTCols);
			System.out.println("This is a special .csv reading version made for Alex-Rowlands dataset!\n"
								+ "parsing .csv file using:\n"
								+ (startTime!=null ? "csvStartTime = " + startTime.toString() + "\n"
								+ "csvSampleRate = " + csvSampleRate + "Hz" : "csvTimeFormat = " + csvTimeFormat.toString())+"\n"
								+ "minimum number of columns per row: " + numColsRequired);
			int xCol = csvXYZTCols.get(0);
			int yCol = csvXYZTCols.get(1);
			int zCol = csvXYZTCols.get(2);
			boolean readTime = false;
			long time = Long.MIN_VALUE;
			if (csvXYZTCols.size()>3) {
				readTime = true;
			} else {
				time = getEpochMillis(startTime);
			}

			while (true) {
				line = accStream.readLine();
				if (lineNumber++ <= csvStartRow) continue;

				if (line==null) {
					accStream.close();
					break;
					// uncomment to loop forever (useful for testing performance)
					// accStream =  new BufferedReader(new FileReader(accFile));
					// continue;
				}
				String[] cols = line.split(csvSplitBy);
				if (cols.length>numColsRequired) {
						if (readTime) {
							String timeStr = "";
							for(int i=3; i<csvXYZTCols.size(); i++) {
								timeStr += cols[csvXYZTCols.get(i)];
								if (i!=csvXYZTCols.size()-1) {
									timeStr += ",";
								}
							}
							time = getEpochMillis(LocalDateTime.parse(timeStr, csvTimeFormat));
							// System.out.println(epochMillisToLocalDateTime(time).toString() + " - " + timeStr);
						}

						double x = Double.parseDouble(cols[xCol]);
						double y = Double.parseDouble(cols[yCol]);
						double z = Double.parseDouble(cols[zCol]);
						epochWriter.newValues(time, x, y, z, 0, new int[] {0});
						if (!readTime) {
							time += 1000/csvSampleRate;
						}
				} else {
					System.err.println(".csv line " + lineNumber + " had too few columns :\n" + line);
				}
			}

 		} catch (Exception excep) {
			excep.printStackTrace(System.err);
			System.err.println("error reading/writing file " + accFile + ": " + excep.toString());
			System.exit(-2);
		}
	}
	/**
	 * Reads a .gt3x file (actually a .zip archive containing at least 3 files).
	 * This method first verifies it is a valid version 1 file (currently V2 not supported)
 	 * it will then parse the header and begin processing the activity.bin file.
	 */
	private static void readG3TXEpochs(String accFile) {

		ZipFile zip = null;
		// readers for the 'activity.bin' & 'info.txt' files inside the .zip
		BufferedReader infoReader = null;
		InputStream activityReader = null;

		try {
			zip = new ZipFile( new File(accFile), ZipFile.OPEN_READ);

			if (!isGT3XV1(zip)) {
				System.err.println("file " + accFile + " is not a V1 g3tx file");
				System.exit(-2);
			}
			for (Enumeration<?> e = zip.entries(); e.hasMoreElements();) {
				ZipEntry entry = (ZipEntry) e.nextElement();
				if (entry.toString().equals("info.txt")) {
					infoReader = new BufferedReader(new InputStreamReader(zip.getInputStream(entry)));
				} else if (entry.toString().equals("activity.bin")) {
					activityReader = zip.getInputStream(entry);
				}
			}

			// underscored are unused for now
			double sampleFreq = -1, accelerationScale = -1, _AccelerationMin, _AccelerationMax;
			long _LastSampleTime, firstSampleTime=-1;
			String serialNumber = "";
			while (infoReader.ready()) {
				String line = infoReader.readLine();
				if (line!=null){
					String[] tokens=line.split(":");
					if ((tokens !=null)  && (tokens.length==2)){
						if (tokens[0].trim().equals("Sample Rate")){
							sampleFreq=Integer.parseInt(tokens[1].trim());
						} else if (tokens[0].trim().equals("Last Sample Time"))
							_LastSampleTime=GT3XfromTickToMillisecond(Long.parseLong(tokens[1].trim()));
						else if (tokens[0].trim().equals("Acceleration Scale"))
							accelerationScale=Double.parseDouble(tokens[1].trim());
						else if (tokens[0].trim().equals("Acceleration Min"))
							_AccelerationMin=Double.parseDouble(tokens[1].trim());
						else if (tokens[0].trim().equals("Acceleration Max"))
							_AccelerationMax=Double.parseDouble(tokens[1].trim());
						else if (tokens[0].trim().equals("Start Date"))
							firstSampleTime=GT3XfromTickToMillisecond(Long.parseLong(tokens[1].trim()));
						else if (tokens[0].trim().equals("Serial Number"))
							serialNumber=tokens[1].trim();
					}
				}
			}

			// Set acceleration scale

			double ACCELERATION_SCALE_FACTOR_NEO_CLE = 341.0; // == 2046 (range of data) / 6 (range of G's)
			double ACCELERATION_SCALE_FACTOR_MOS = 256.0; // == 2048/8?

			if((serialNumber.startsWith("NEO") || (serialNumber.startsWith("CLE")))) {
				accelerationScale = ACCELERATION_SCALE_FACTOR_NEO_CLE;
			} else if(serialNumber.startsWith("MOS")){
				accelerationScale = ACCELERATION_SCALE_FACTOR_MOS;
			}

			if (sampleFreq==-1 || accelerationScale==-1 || firstSampleTime==-1) {
				System.err.println("error parsing "+accFile+", info.txt must contain 'Sample Rate', ' Start Date', and (usually) 'Acceleration Scale'.");
				System.exit(-2);
			}
			System.out.println("sampleFreq:" + sampleFreq);
			double sampleDelta_old = Math.round(1000.0/sampleFreq * 100d) / 100d;  // round the delta to its fourth decimal
			System.out.println("sampleDelta before rounding:" + sampleDelta_old);
			double sampleDelta = 1000.0/sampleFreq;  // don't round the delta to its fourth decimal
			System.out.println("sampleDelta after rounding:" + sampleDelta);
//			sampleDelta = Math.round(1000.0/sampleFreq * 100d) / 100d;  // round the delta to its fourth decimal

			// else leave as specified in info.txt?
			readG3TXEpochPairs(activityReader, sampleDelta, sampleFreq, accelerationScale, firstSampleTime);
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
	 ** Helper method that converts .NET ticks that Actigraph GT3X uses to millisecond (local)
	 ** method from: https://github.com/SPADES-PUBLIC/mHealth-GT3X-converter-public/blob/master/src/com/qmedic/data/converter/gt3x/GT3XUtils.java
	 **/
	private static long GT3XfromTickToMillisecond(final long ticks)
	{
		Date date = new Date((ticks - 621355968000000000L) / 10000);
		return date.getTime();
	}

	// Convert LocalDateTime to epoch milliseconds (from 1970 epoch)
	private static long getEpochMillis(LocalDateTime date) {
		return date.toInstant(ZoneOffset.UTC).toEpochMilli();
	}

	/**
	 ** Method to read all the x/y/z data from a GT3X (V1) activity.bin file.
	 ** File specification at: https://github.com/actigraph/NHANES-GT3X-File-Format/blob/master/fileformats/activity.bin.md
	 ** Data is stored sequentially at the sample rate specified in the header (1/f = sampleDelta in milliseconds)
	 ** Each pair of readings occupies an awkward 9 bytes to conserve space, so must be read 2 at a time.
	 ** The readings should range from -2046 to 2046, covering -6 to 6 G's,
	 ** thus the maximum accuracy is 0.003 G's. The values -2048, -2047 & 2047 should never appear in the stream.
	 **/
	private static void readG3TXEpochPairs(
			InputStream activityReader,
			double sampleDelta,
			double sampleFreq,
			double accelerationScale,
			long firstSampleTime // in milliseconds
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
					epochWriter.newValues(time, x, y, z, temp, errCounter);

					samples += 1;

				}
			}
		}
		catch (IOException ex) {
			System.out.println("End of .g3tx file reached");
		}
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


	/*
	 * This method checks if a file is of GT3X format version 1 It returns true
	 * if the file is of the correct format otherwise it returns false
	 */
	private static boolean isGT3XV1(final ZipFile zip) throws IOException {

		// Check if the file contains the necessary Actigraph files
		boolean hasActivityData = false;
		boolean hasLuxData = false;
		boolean hasInfoData = false;
		for (Enumeration<?> e = zip.entries(); e.hasMoreElements();) {
			ZipEntry entry = (ZipEntry) e.nextElement();
			if (entry.toString().equals("activity.bin"))
				hasActivityData = true;
			if (entry.toString().equals("lux.bin"))
				hasLuxData = true;
			if (entry.toString().equals("info.txt"))
				hasInfoData = true;
		}

		if (hasActivityData && hasLuxData && hasInfoData)
			return true;

		return false;
	}



	/**
	 * Read and process Axivity CWA file. Setup file reading infrastructure
	 * and then call readCwaBuffer() method
	**/
	private static void readCwaEpochs(String accFile, int timeZoneOffset) {

		int[] errCounter = new int[] { 0 }; // store val if updated in other
											// method
		// Inter-block timstamp tracking
		LocalDateTime[] lastBlockTime = { null };
		int[] lastBlockTimeIndex = { 0 };

		// data block support variables
		String header = "";

		// Variables for tracking start offset of header
		LocalDateTime SESSION_START = null;
		long START_OFFSET_NANOS = 0;


		int bufSize = 512;
		ByteBuffer buf = ByteBuffer.allocate(bufSize);
		try ( FileInputStream accStream = new FileInputStream(accFile); ) {
			FileChannel rawAccReader = accStream.getChannel();

			// now read every page in CWA file
			int pageCount = 0;
			long memSizePages = rawAccReader.size() / bufSize;
			boolean USE_PRECISE_TIME = true; // true uses block fractional time
												// and
												// interpolates timestamp
												// between blocks.
			while (rawAccReader.read(buf) != -1) {
				readCwaBuffer(buf, SESSION_START, START_OFFSET_NANOS,
					USE_PRECISE_TIME, lastBlockTime, lastBlockTimeIndex, header,
					errCounter, timeZoneOffset);
				buf.clear();
				// option to provide status update to user...
				pageCount++;
				if (verbose && pageCount % 10000 == 0) {
					System.out.print((pageCount * 100 / memSizePages) + "%\t");
				}
			}
			rawAccReader.close();
		} catch (Exception excep) {
			excep.printStackTrace(System.err);
			System.err.println("error reading/writing file " + accFile + ": " + excep.toString());
			System.exit(-2);
		}
	}

	/**
	 * Read and process Axivity CWA.gz gzipped file. Setup file reading
	 * infrastructure and then call readCwaBuffer() method
	**/
	private static void readCwaGzEpochs(String accFile, int timeZoneOffset) {

        int[] errCounter = new int[] { 0 }; // store val if updated in other
                                            // method
        // Inter-block timstamp tracking
        LocalDateTime[] lastBlockTime = { null };
        int[] lastBlockTimeIndex = { 0 };

        // data block support variables
        String header = "";

        // Variables for tracking start offset of header
        LocalDateTime SESSION_START = null;
        long START_OFFSET_NANOS = 0;


        int bufSize = 512;
        ByteBuffer buf = ByteBuffer.allocate(bufSize);
        try ( FileInputStream accStream = new FileInputStream(accFile); ) {
            GZIPInputStream in = new GZIPInputStream(accStream);
            ReadableByteChannel rawAccReader = Channels.newChannel(in);

            // now read every page in CWA file
            int pageCount = 0;
            long memSizePages = getUncompressedSizeofGzipFile(accFile) / bufSize;
            boolean USE_PRECISE_TIME = true; // true uses block fractional time
                                                // and
                                                // interpolates timestamp
                                                // between blocks.
            while (rawAccReader.read(buf) != -1) {
                readCwaBuffer(buf, SESSION_START, START_OFFSET_NANOS,
                    USE_PRECISE_TIME, lastBlockTime, lastBlockTimeIndex, header,
                    errCounter, timeZoneOffset);
                buf.clear();
                // option to provide status update to user...
                pageCount++;
                if (verbose && pageCount % 10000 == 0) {
                    System.out.print((pageCount * 100 / memSizePages) + "%\t");
                }
            }
            rawAccReader.close();
        } catch (Exception excep) {
            excep.printStackTrace(System.err);
            System.err.println("error reading/writing file " + accFile + ": " + excep.toString());
            System.exit(-2);
        }
    }

    private static long getUncompressedSizeofGzipFile(String gzipFile){
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

	/**
	 * Read Axivity CWA file, then call method to write epochs from raw data.
	 * Epochs will be written to epochFileWriter.
	 * Read data block HEX values, store each raw reading, then continually test
	 * if an epoch of data has been collected or not. Finally, write each epoch
	 * to epochFileWriter. CWA format is described at:
     * https://github.com/digitalinteraction/openmovement/blob/master/Downloads/AX3/AX3-CWA-Format.txt
	**/
	private static void readCwaBuffer(ByteBuffer buf, LocalDateTime SESSION_START,
		long START_OFFSET_NANOS, boolean USE_PRECISE_TIME,
		LocalDateTime[] lastBlockTime, int[] lastBlockTimeIndex, String header,
		int[] errCounter, int timeZoneOffset)
	{
		buf.flip();
		buf.order(ByteOrder.LITTLE_ENDIAN);
		header = (char) buf.get() + "";
		header += (char) buf.get() + "";
		if (header.equals("MD")) {
			// Read first page (& data-block) to get time, temp,
			// measureFreq
			// start-epoch values
			try {
				SESSION_START = cwaHeaderLoggingStartTime(buf, timeZoneOffset);
				System.out.println("Session start:" + SESSION_START);
			} catch (Exception e) {
				System.err.println("No preset start time");
			}
		} else if (header.equals("AX")) {
			// read each individual page block, and process epochs...
			try {

				// read block header items
				long blockTimestamp = getUnsignedInt(buf, 14);
				int light = getUnsignedShort(buf, 18);
				double temperature = (getUnsignedShort(buf, 20) * 150.0 - 20500) / 1000;
				short rateCode = (short) (buf.get(24) & 0xff);
				short numAxesBPS = (short) (buf.get(25) & 0xff);
				int sampleCount = getUnsignedShort(buf, 28);
				short timestampOffset = 0;
				double sampleFreq = 0;
				int fractional = 0; // 1/65536th of a second fractions

				// check not very old file as pos 26=freq rather than
				// timestamp offset
				if (rateCode != 0) {
					timestampOffset = buf.getShort(26); // timestamp
														// offset ok
					// if fractional offset, then timestamp offset was
					// artificially
					// modified for backwards-compatibility ...
					// therefore undo this...
					int oldDeviceId = getUnsignedShort(buf, 4);
					if ((oldDeviceId & 0x8000) != 0) {
						sampleFreq = 3200.0 / (1 << (15 - (rateCode & 15)));
						if (USE_PRECISE_TIME) {
							// Need to undo backwards-compatible shim:
							// Take into account how many whole samples
							// the fractional part of timestamp
							// accounts for:
							// relativeOffset = fifoLength -
							// (short)(((unsigned long)timeFractional *
							// AccelFrequency()) >> 16);
							// nearest whole sample
							// whole-sec | /fifo-pos@time
							// | |/
							// [0][1][2][3][4][5][6][7][8][9]
							// use 15-bits as 16-bit fractional time
							fractional = ((oldDeviceId & 0x7fff) << 1);
							// frequency is truncated to int in firmware
							timestampOffset += ((fractional * (int) sampleFreq) >> 16);
						}
					}
				} else {
					sampleFreq = buf.getShort(26);
					// very old format, where pos26 = freq
				}

				// calculate num bytes per sample...
				byte bytesPerSample = 4;
				int NUM_AXES_PER_SAMPLE = 3;
				if ((numAxesBPS & 0x0f) == 2) {
					bytesPerSample = 6; // 3*16-bit
				} else if ((numAxesBPS & 0x0f) == 0) {
					bytesPerSample = 4; // 3*10-bit + 2
				}

				// Limit values
				int maxSamples = 480 / bytesPerSample; // 80 or 120
														// samples/block
				if (sampleCount > maxSamples) {
					sampleCount = maxSamples;
				}
				if (sampleFreq <= 0) {
					sampleFreq = 1;
				}

				// determine time for indexed sample within block
				LocalDateTime blockTime = getCwaTimestamp((int) blockTimestamp, fractional, timeZoneOffset);
				// first & last sample. Actually, last = first sample in
				// next block
				LocalDateTime firstSampleTime, lastSampleTime;
				// if no interval between times (or interval too large)
				long spanToSample = 0;
				if (lastBlockTime[0] != null) {
					spanToSample = Duration.between(lastBlockTime[0], blockTime).toNanos();
				}
				if (!USE_PRECISE_TIME ||
						lastBlockTime[0] == null ||
						timestampOffset <= lastBlockTimeIndex[0] ||
						spanToSample <= 0 ||
						spanToSample > 1000000000.0 * 2 * maxSamples / sampleFreq
					) {
					float offsetStart = (float) -timestampOffset / (float) sampleFreq;
					firstSampleTime = blockTime.plusNanos(secs2Nanos(offsetStart));
					lastSampleTime = firstSampleTime.plusNanos(secs2Nanos(sampleCount / sampleFreq));
				} else {
					double gap = (double) spanToSample / (-lastBlockTimeIndex[0] + timestampOffset);
					firstSampleTime = lastBlockTime[0].plusNanos((long) (-lastBlockTimeIndex[0] * gap));
					lastSampleTime = lastBlockTime[0]
							.plusNanos((long) ((-lastBlockTimeIndex[0] + sampleCount) * gap));
				}


				// Last block time
				lastBlockTime[0] = blockTime;
				// Advance last block time index for next block
				lastBlockTimeIndex[0] = timestampOffset - sampleCount;
				// Overall span of block
				long spanNanos = Duration.between(firstSampleTime, lastSampleTime).toNanos();

				// raw reading values
				long value = 0; // x/y/z vals
				short xRaw = 0;
				short yRaw = 0;
				short zRaw = 0;
				double x = 0.0;
				double y = 0.0;
				double z = 0.0;

				// loop through each line in data block and check if it
				// is last in
				// epoch then write epoch summary to file
				// an epoch will have a start+end time, and fixed
				// duration
				for (int i = 0; i < sampleCount; i++) {
					if (USE_PRECISE_TIME) {
						// Calculate each sample's time, not successively adding
						// so that we don't accumulate any errors
						blockTime = firstSampleTime.plusNanos((long) (i * (double) spanNanos / sampleCount));
					} else {

						if (i == 0) {
							blockTime = firstSampleTime; // emulate original
															// behaviour
						} else {
							blockTime = blockTime.plusNanos(secs2Nanos(1.0 / sampleFreq));
						}
					}

					if (bytesPerSample == 4) {
						try {
							value = getUnsignedInt(buf, 30 + 4 * i);
						} catch (Exception excep) {
							errCounter[0] += 1;
							System.err.println("xyz reading err: " + excep.toString());
							break; // rest of block/page may be
									// corrupted
						}
						// Sign-extend 10-bit values, adjust for
						// exponents
						xRaw = (short) ((short) (0xffffffc0 & (value << 6)) >> (6 - ((value >> 30) & 0x03)));
						yRaw = (short) ((short) (0xffffffc0 & (value >> 4)) >> (6 - ((value >> 30) & 0x03)));
						zRaw = (short) ((short) (0xffffffc0 & (value >> 14)) >> (6 - ((value >> 30) & 0x03)));
					} else if (bytesPerSample == 6) {
						try {
							xRaw = buf.getShort(30 + 2 * NUM_AXES_PER_SAMPLE * i + 0);
							yRaw = buf.getShort(30 + 2 * NUM_AXES_PER_SAMPLE * i + 2);
							zRaw = buf.getShort(30 + 2 * NUM_AXES_PER_SAMPLE * i + 4);
						} catch (Exception excep) {
							errCounter[0] += 1;
							System.err.println("xyz read err: " + excep.toString());
							break; // rest of block/page may be
									// corrupted
						}
					} else {
						xRaw = 0;
						yRaw = 0;
						zRaw = 0;
					}
					x = xRaw / 256.0;
					y = yRaw / 256.0;
					z = zRaw / 256.0;

					epochWriter.newValues(getEpochMillis(blockTime.plusNanos(START_OFFSET_NANOS)), x, y, z, temperature, errCounter);

				}
			} catch (Exception excep) {
				excep.printStackTrace(System.err);
				System.err.println("block err @ " + lastBlockTime[0].toString() + ": " + excep.toString());
			}
		}
	}



	// Parse HEX values, CWA format is described at:
    // https://github.com/digitalinteraction/openmovement/blob/master/Downloads/AX3/AX3-CWA-Format.txt
	private static LocalDateTime getCwaTimestamp(int cwaTimestamp, int fractional, int timeZoneOffset) {
		LocalDateTime tStamp;
		int year = (int) ((cwaTimestamp >> 26) & 0x3f) + 2000;
		int month = (int) ((cwaTimestamp >> 22) & 0x0f);
		int day = (int) ((cwaTimestamp >> 17) & 0x1f);
		int hours = (int) ((cwaTimestamp >> 12) & 0x1f);
		int mins = (int) ((cwaTimestamp >> 6) & 0x3f);
		int secs = (int) ((cwaTimestamp) & 0x3f);
		tStamp = LocalDateTime.of(year, month, day, hours, mins, secs);
		tStamp = tStamp.plusMinutes(timeZoneOffset);
		// add 1/65536th fractions of a second
		tStamp = tStamp.plusNanos(secs2Nanos(fractional / 65536.0));
		return tStamp;
	}

	private static LocalDateTime epochMillisToLocalDateTime(long m) {
		return LocalDateTime.ofEpochSecond((long) Math.floor(m/1000), (int) TimeUnit.MILLISECONDS.toNanos((m % 1000)), ZoneOffset.UTC);
	}

	private static LocalDateTime cwaHeaderLoggingStartTime(ByteBuffer buf, int timeZoneOffset) {
		long delayedLoggingStartTime = getUnsignedInt(buf, 13);
		return getCwaTimestamp((int) delayedLoggingStartTime, 0, timeZoneOffset);
	}

	// credit for next 2 methods goes to:
	// http://stackoverflow.com/questions/9883472/is-it-possiable-to-have-an-unsigned-bytebuffer-in-java
	private static long getUnsignedInt(ByteBuffer bb, int position) {
		return ((long) bb.getInt(position) & 0xffffffffL);
	}

	private static int getUnsignedShort(ByteBuffer bb, int position) {
		return (bb.getShort(position) & 0xffff);
	}

	/**
	 * Read GENEA bin file pages, then call method to write epochs from raw
	 * data. Epochs will be written to epochFileWriter.
	 */
	private static void readGeneaEpochs(String accFile) {

		int fileHeaderSize = 59;
		int linesToAxesCalibration = 47;
		int pageHeaderSize = 9;
		// epoch creation support variables
		int[] errCounter = new int[] { 0 }; // store val if updated in other
											// method

		try {
			BufferedReader rawAccReader = new BufferedReader(new FileReader(accFile));
			// Read header to determine mfrGain and mfrOffset values
			double[] mfrGain = new double[3];
			int[] mfrOffset = new int[3];
			// memory size in pages
			int memSizePages = parseBinFileHeader(rawAccReader, fileHeaderSize, linesToAxesCalibration, mfrGain, mfrOffset);

			int pageCount = 1;
			String header;
			LocalDateTime blockTime = LocalDateTime.of(1999, 1, 1, 1, 1, 1);
			double temperature = 0.0;
			double sampleFreq = 0.0;
			String dataBlock;
			String timeFmtStr = "yyyy-MM-dd HH:mm:ss:SSS";
			DateTimeFormatter timeFmt = DateTimeFormatter.ofPattern(timeFmtStr);
			while ((readLine(rawAccReader)) != null) {
				// header: "Recorded Data" (0), serialCode (1), seq num (2),
				// pageTime (3), unassigned (4), temp (5), batteryVolt (6),
				// deviceStatus (7), sampleFreq (8),
				// Then: dataBlock (9)
				// line "page = readLine(..." above will read 1st header line
				// (c=0)
				for (int c = 1; c < pageHeaderSize; c++) {
					try {
						header = readLine(rawAccReader);
						if (c == 3) {
							blockTime = LocalDateTime.parse(header.split("Time:")[1], timeFmt);
						} else if (c == 5) {
							temperature = Double.parseDouble(header.split(":")[1]);
						} else if (c == 8) {
							sampleFreq = Double.parseDouble(header.split(":")[1]);
						}
					} catch (Exception excep) {
						System.err.println(excep.toString());
						continue; // to keep reading sequence correct
					}
				}

				// now process hex dataBlock
				dataBlock = readLine(rawAccReader);


				// raw reading values
				int hexPosition = 0;
				int xRaw = 0;
				int yRaw = 0;
				int zRaw = 0;
				double x = 0.0;
				double y = 0.0;
				double z = 0.0;

				// loop through each reading in data block and check if it is
				// last in
				// epoch then write epoch summary to file
				// an epoch will have a start+end time, and fixed duration
				while (hexPosition < dataBlock.length()) {
					try {
						xRaw = getSignedIntFromHex(dataBlock, hexPosition, 3);
						yRaw = getSignedIntFromHex(dataBlock, hexPosition + 3, 3);
						zRaw = getSignedIntFromHex(dataBlock, hexPosition + 6, 3);
					} catch (Exception excep) {
						errCounter[0] += 1;
						System.err.println("block err @ " + blockTime.toString() + ": " + excep.toString());
						break; // rest of block/page could be corrupted
					}
					// todo *** read in light[36:46] (10 bits to signed int) and
					// button[47] (bool) values...

					// update values to calibrated measure (taken from GENEActiv
					// manual)
					x = (xRaw * 100.0d - mfrOffset[0]) / mfrGain[0];
					y = (yRaw * 100.0d - mfrOffset[1]) / mfrGain[1];
					z = (zRaw * 100.0d - mfrOffset[2]) / mfrGain[2];

					epochWriter.newValues(getEpochMillis(blockTime), x, y, z, temperature, errCounter);


					hexPosition += 12;
					blockTime = blockTime.plusNanos(secs2Nanos(1.0 / sampleFreq));
				}
				// option to provide status update to user...
				pageCount++;
				if (verbose && pageCount % 10000 == 0) {
					System.out.print((pageCount * 100 / memSizePages) + "%\t");
				}
			}
			rawAccReader.close();
		} catch (Exception excep) {
			excep.printStackTrace(System.err);
			System.err.println("error reading/writing file " + accFile + ": " + excep.toString());
			System.exit(-2);
		}
	}

	/**
	 * Replicates bin file header to epoch file, also calculates and returns
	 * x/y/z gain/offset values along with number of pages of data in file bin
	 * format described in GENEActiv manual ("Decoding .bin files", pg.27)
	 * http://www.geneactiv.org/wp-content/uploads/2014/03/
	 * geneactiv_instruction_manual_v1.2.pdf
	 */
	private static int parseBinFileHeader(BufferedReader reader, int fileHeaderSize, int linesToAxesCalibration,
			double[] gainVals, int[] offsetVals) {
		// read first c lines in bin file to writer
		for (int c = 0; c < linesToAxesCalibration; c++) {
			readLine(reader);
		}
		// read axes calibration lines for gain and offset values
		// data like -> x gain:25548 \n x offset:574 ... Volts:300 \n Lux:800
		gainVals[0] = Double.parseDouble(readLine(reader).split(":")[1]); // xGain
		offsetVals[0] = Integer.parseInt(readLine(reader).split(":")[1]); // xOffset
		gainVals[1] = Double.parseDouble(readLine(reader).split(":")[1]); // y
		offsetVals[1] = Integer.parseInt(readLine(reader).split(":")[1]); // y
		gainVals[2] = Double.parseDouble(readLine(reader).split(":")[1]); // z
		offsetVals[2] = Integer.parseInt(readLine(reader).split(":")[1]); // z
		int volts = Integer.parseInt(readLine(reader).split(":")[1]); // volts
		int lux = Integer.parseInt(readLine(reader).split(":")[1]); // lux
		readLine(reader); // 9 blank
		readLine(reader); // 10 memory status header
		int memorySizePages = Integer.parseInt(readLine(reader).split(":")[1]); // 11

		// ignore remaining header lines in bin file
		for (int c = 0; c < fileHeaderSize - linesToAxesCalibration - 11; c++) {
			readLine(reader);
		}
		return memorySizePages;
	}

	private static int getSignedIntFromHex(String dataBlock, int startPos, int length) {
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

	private static long secs2Nanos(double num) {
		return (long) (TimeUnit.SECONDS.toNanos(1) * num);
	}

	private static String readLine(BufferedReader fReader) {
		String line = "";
		try {
			line = fReader.readLine();
		} catch (Exception excep) {
			System.err.println(excep.toString());
		}
		return line;
	}

	private static void writeLine(BufferedWriter fileWriter, String line) {
		try {
			fileWriter.write(line + "\n");
		} catch (Exception excep) {
			System.err.println("line write error: " + excep.toString());
		}
	}

}
