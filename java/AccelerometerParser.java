
//BSD 2-Clause (c) 2014: A.Doherty (Oxford), D.Jackson, N.Hammerla (Newcastle)
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.time.format.DateTimeFormatter;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.LinkedList;
import java.util.List;
import java.util.TimeZone;


/**
 * Calculates epoch summaries from an AX3 .CWA file. Class/application can be
 * called from the command line as follows: java AxivityAx3Epochs inputFile.CWA
 */
public class AccelerometerParser {

	private static final DecimalFormat DF6 = new DecimalFormat("0.000000");
	private static final DecimalFormat DF3 = new DecimalFormat("0.000");
	private static final DecimalFormat DF2 = new DecimalFormat("0.00");
	

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
		boolean verbose = false; //to facilitate logging

		DF6.setRoundingMode(RoundingMode.CEILING);
		DF2.setRoundingMode(RoundingMode.CEILING);
    	DF3.setRoundingMode(RoundingMode.HALF_UP); // To match their implementation

    	// epochWriter parameters
    	int epochPeriod = 30;
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
    	int numFFTbins = 15; // number of fft bins to print

    	// Must supply additional information when loading from a .csv file
    	LocalDateTime csvStartTime = null; // start date of first sample
    	double csvSampleRate = -1; // must specify sample rate if time column and date format not given
    	int csvStartRow = 0;
    	// [0, 1, 2] = assume X is 1st column, Y is 2nd, Z is 3rd, and no time column.
    	// if list is longer than 3 elements the extra elements are the columns that make up a time value
    	// e.g. [0,1,2,3,6] means use 1st three columns for X/Y/Z, and concatenate cols 3 & 6 to parse time
    	// (useful when time and date are in separate columns)
    	List<Integer> csvXYZTCols = Arrays.asList( 0,1,2 );


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
				} else if (funcName.equals("numFFTbins")) {
					numFFTbins = Integer.parseInt(funcParam);
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

		EpochWriter epochWriter = null;
		try {
			System.out.println("Intermediate file: " + outputFile);
   			epochWriter = DeviceReader.setupEpochWriter(
   				outputFile, useFilter, rawOutput, rawFile, fftOutput, npyOutput,
        		npyFile, getAxisMeans, getSanDiegoFeatures, getMADFeatures, getUnileverFeatures,
        		get3DFourier, getEachAxis, numFFTbins, useAbs, timeFormat,
        		epochPeriod, sampleRate, range, swIntercept, swSlope, tempCoef,
        		meanTemp, getStationaryBouts, stationaryStd, getEpochCovariance,
        		startTime, endTime, verbose
   				);

			// process file if input parameters are all ok
			if (accFile.toLowerCase().endsWith(".cwa")) {
				AxivityReader.readCwaEpochs(accFile, epochWriter, 
											timeZoneOffset, verbose);
			} else if (accFile.toLowerCase().endsWith(".cwa.gz")) {
                AxivityReader.readCwaGzEpochs(accFile, epochWriter, 
                								timeZoneOffset, verbose);
            } else if (accFile.toLowerCase().endsWith(".bin")) {
				GENEActivReader.readGeneaEpochs(accFile, epochWriter, verbose);
			} else if (accFile.toLowerCase().endsWith(".gt3x")) {
				ActigraphReader.readG3TXEpochs(accFile, epochWriter, verbose);
			} else if (accFile.toLowerCase().endsWith(".csv")) {
				if ((csvStartTime==null || csvSampleRate<=0) && (csvTimeFormat==null || csvXYZTCols.size()<=3)) {
					System.err.println("Must specify either 'csvStartTime' (YYYY-MM-DDThh:mm format) and 'csvSampleRate', or:\n"
									 + "'csvXYZYCols' (at least 4 values) and 'csvTimeFormat'\n"
									 + "options for .csv type file: " + accFile);
					System.exit(-1);
				}
				CsvReader.readCSVEpochs(accFile, epochWriter, csvStartTime,
					csvSampleRate, csvTimeFormat, csvStartRow, csvXYZTCols);
			} else {
				System.err.println("Unrecognised file format for: " + accFile);
				System.exit(-1);
			}
		} catch (Exception excep) {
			excep.printStackTrace(System.err);
			System.err.println("error reading/writing file " + outputFile + ": " 
								+ excep.toString());
			System.exit(-2);
		} finally {
			try {
				epochWriter.closeWriters();
			} catch (Exception ex) {
				/* ignore */
			}
		}

		// if no errors then return success code
		System.exit(0);
	}


	private static LocalDateTime epochMillisToLocalDateTime(long m) {
		return LocalDateTime.ofEpochSecond((long) Math.floor(m/1000), 
			(int) TimeUnit.MILLISECONDS.toNanos((m % 1000)), 
			ZoneOffset.UTC);
	}
	

}