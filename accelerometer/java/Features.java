/*
 *
 * Code for extracting features from raw triaxial acceleration measurements.
 * The features are derived from four papers:
 *
 * Hip and Wrist Accelerometer Algorithms for Free-Living Behavior
 * Classification. Ellis K, Kerr J, Godbole S, Staudenmayer J, Lanckriet G.
 *
 * A universal, accurate intensity-based classification of different
 * physical activities using raw data of accelerometer. Henri Vaha-Ypya,
 * Tommi Vasankari, Pauliina Husu, Jaana Suni and Harri Sievanen
 *
 * Physical Activity Classification using the GENEA Wrist Worn
 * Accelerometer Shaoyan Zhang, Alex V. Rowlands, Peter Murray, Tina Hurst
 *
 * Activity recognition using a single accelerometer placed at the wrist or ankle.
 * Mannini A, Intille SS, Rosenberger M, Sabatini AM, Haskell W.
 *
 */
import org.jtransforms.fft.DoubleFFT_1D;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
public class Features {

    public static double[] getFeatures(
            double[] x,
            double[] y,
            double[] z,
            double[] filteredVM,
            int sampleRate){

        // get San Diego (Ellis) features
        double[] sanDiegoFeats = calculateSanDiegoFeatures(x, y, z, sampleRate);

        // get MAD features
        double[] madFeats = calculateMADFeatures(x, y, z);

        // get arm angel features
        double[] armFeats = calculateArmFeatures(x, y, z, sampleRate);

        // get Unilever (Zhang/Rowlands) features
        double[] uniFeats = unileverFeatures(filteredVM, sampleRate);

        // construct final output array
        // don't forget to change header method immediately below !!!
        double[] output = AccStats.combineArrays(sanDiegoFeats, madFeats);
        output = AccStats.combineArrays(output, armFeats);
        output = AccStats.combineArrays(output, uniFeats);

        return output;
        // todo method to write header line order (use static class variables )
    }

    public static String getFeaturesHeader() {
        String header = getSanDiegoFeaturesHeader();
        header += "," + getMADFeaturesHeader();
        header += "," + getArmFeaturesHeader();
        header += "," + getUnileverFeaturesHeader();
        return header;
    }


    private static double[] calculateSanDiegoFeatures(double[] x, double[] y, double[] z, int sampleRate) {

        /*
            This function aims to replicate the following R-code:

              # v = vector magnitude
              v = sqrt(rowSums(w ^ 2))

              fMean = mean(v)
              fStd = sd(v)
              if (fMean > 0) {
                fCoefVariation = fStd / fMean
              } else {
                fCoefVariation = 0
              }
              fMedian = median(v)
              fMin = min(v)
              fMax = max(v)
              f25thP = quantile(v, 0.25)[[1]]
              f75thP = quantile(v, 0.75)[[1]]

        */

        int n = x.length;

        // San Diego g values
        // the g matric contains the estimated gravity vector
        // this is essentially a low pass filter
        double[] gg = sanDiegoGetAvgGravity(x, y, z, sampleRate);
        double gxMean = gg[0];
        double gyMean = gg[1];
        double gzMean = gg[2];


        // subtract column means and get vector magnitude
        double[] v = new double[n]; // vector magnitude
        double[] wx = new double[n]; // gravity adjusted weights
        double[] wy = new double[n];
        double[] wz = new double[n];
        for (int i = 0; i < n; i++) {
            wx[i] = x[i]-gxMean;
            wy[i] = y[i]-gyMean;
            wz[i] = z[i]-gzMean;
            v[i] = AccStats.getVectorMagnitude( wx[i], wy[i], wz[i]);
        }

        // Write epoch
        double sdMean = AccStats.mean(v);
        double sdStd = AccStats.stdR(v, sdMean);
        double sdCoefVariation = 0.0;
        if (sdMean!=0) sdCoefVariation = sdStd/sdMean;
        double[] paQuartiles = AccStats.percentiles(v, new double[] {0, 0.25, 0.5, 0.75, 1});

        //correlations
        double autoCorrelation = correlation(v, v, sampleRate);
        double xyCorrelation = correlation(wx, wy);
        double xzCorrelation = correlation(wx, wz);
        double yzCorrelation = correlation(wy, wz);

        // Roll, Pitch, Yaw
        double [] angleAvgStdYZ = AccStats.angleAvgStd(wy, wz); //roll
        double [] angleAvgStdZX = AccStats.angleAvgStd(wz, wx); //pitch
        double [] angleAvgStdYX = AccStats.angleAvgStd(wy, wx); //yaw

        // gravity component angles
        double gxyAngle = Math.atan2(gyMean,gzMean);
        double gzxAngle = Math.atan2(gzMean,gxMean);
        double gyxAngle = Math.atan2(gyMean,gxMean);

        // don't forget to change header method immediately below !!!
        double[] output = new double[]{
            sdMean,
            sdStd,
            sdCoefVariation,
            paQuartiles[2], // median
            paQuartiles[0], // min
            paQuartiles[4], // max
            paQuartiles[1], // 25th
            paQuartiles[3], // 75th
            autoCorrelation,
            xyCorrelation,
            xzCorrelation,
            yzCorrelation,
            angleAvgStdYZ[0], // mean roll
            angleAvgStdZX[0], // mean pitch
            angleAvgStdYX[0], // mean yaw
            angleAvgStdYZ[1], // sd roll
            angleAvgStdZX[1], // sd pitch
            angleAvgStdYX[1], // sd yaw
            gxyAngle,
            gzxAngle,
            gyxAngle,
        };
        double[] fftVals = sanDiegoFFT(v, sampleRate);
        return AccStats.combineArrays(output, fftVals);

    }

    private static String getSanDiegoFeaturesHeader() {
        String header = "mean,sd,coefvariation";
        header += ",median,min,max,25thp,75thp";
        header += ",autocorr,corrxy,corrxz,corryz";
        header += ",avgroll,avgpitch,avgyaw";
        header += ",sdroll,sdpitch,sdyaw";
        header += ",rollg,pitchg,yawg";
        header += "," + getSanDiegoFFTHeader();
        return header;
    }


    // returns { x, y, z } averages of gravity
    private static double[] sanDiegoGetAvgGravity(
        double[] x,
        double[] y,
        double[] z,
        int sampleRate) {
        // San Diego paper 0.5Hz? low-pass filter approximation
        // this code takes in w and returns gg
        /*    R-code: (Fs is sampleRate, w are x/y/z values)

              g = matrix(0, nrow(w), 3)
              x = 0.9
              g[1, ] = (1-x) * w[1, ]
              for (n in 2:nrow(w)) {
                g[n, ] = x * g[n-1] + (1-x) * w[n, ]
              }
              g = g[Fs:nrow(g), ] # ignore beginning
              gg = colMeans(g)
              w = w - gg
        */
        int n = x.length;
        // number of moving average values to estimate gravity direction with
        int gn = n-(sampleRate-1);
        int gStartIdx = n - gn; // number of gravity values to discard at beginning

        double[] gx = new double[gn];
        double[] gy = new double[gn];
        double[] gz = new double[gn];

        {
            // calculating moving average of signal
            double weight = 0.9;
            double xMovAvg = (1-weight)*x[0];
            double yMovAvg = (1-weight)*y[0];
            double zMovAvg = (1-weight)*z[0];

            for (int i = 1; i < n; i++) {
                xMovAvg = xMovAvg * weight + (1-weight) * x[i];
                yMovAvg = yMovAvg * weight + (1-weight) * y[i];
                zMovAvg = zMovAvg * weight + (1-weight) * z[i];
                if (i>= gStartIdx){
                    gx[i-gStartIdx] = xMovAvg;
                    gy[i-gStartIdx] = yMovAvg;
                    gz[i-gStartIdx] = zMovAvg;
                }
            }
        }

        // column means
        double gxMean = AccStats.mean(gx);
        double gyMean = AccStats.mean(gy);
        double gzMean = AccStats.mean(gz);

        return new double[] {gxMean, gyMean, gzMean};
    }


    private static double[] sanDiegoFFT(double[] v, int sampleRate)
    {
        final int n = v.length;
        final double vMean = AccStats.mean(v);

        // Initialize array to compute FFT coefs
        double[] vFFT = new double[n];
        for (int i = 0; i < n; i++){
            vFFT[i] = v[i] - vMean;  // note: we remove the 0Hz freq
        }

        HanningWindow(vFFT, vFFT.length);
        new DoubleFFT_1D(vFFT.length).realForward(vFFT);  // FFT library computes coefs inplace
        final double[] vFFTpow = getFFTpower(vFFT);  // parse FFT coefs to obtain the powers

        /*
        Compute spectral entropy
        See https://www.mathworks.com/help/signal/ref/pentropy.html#mw_a57f549d-996c-47d9-8d45-e80cb739ed41
        */
        double spectralEntropy = 0.0;  // spectral entropy
        final double vFFTpowsum = AccStats.sum(vFFTpow);
        for (int i = 0; i < vFFTpow.length; i++) {
            double p = vFFTpow[i] / (vFFTpowsum + 1E-8);
            if (p <= 0) continue;  // skip to next loop if power is non-positive
            spectralEntropy += -p * Math.log(p + 1E-8);
        }
        spectralEntropy /= Math.log(vFFTpow.length);  // Normalize spectral entropy

        //Find dominant frequencies overall, also between 0.3Hz and 3Hz
        final double FFTinterval = sampleRate / (1.0 * n); // (Hz)
        double f1=-1, f33=-1;
        double p1=0, p33=0;
        for (int i = 0; i < vFFTpow.length; i++) {
            double freq = FFTinterval * i;
            double p = vFFTpow[i];
            if (p > p1) {
                f1 = freq;
                p1 = p;
            }
            if (freq > 0.3 && freq < 3 && p > p33) {
                f33 = freq;
                p33 = p;
            }
        }
        // Use logscale for convenience as these tend to be very large
        p1 = Math.log(p1 + 1E-8);
        p33 = Math.log(p33 + 1E-8);

        /*
        Estimate powers for frequencies 0-9 using Welch's method
        See: https://en.wikipedia.org/wiki/Welch%27s_method
        Note: Using the average magnitudes (instead of powers) yielded
        slightly better classification results in random forest
        */
        final int numBins = 10;
        double[] binnedFFT = new double[numBins];
        for (int i = 0; i < numBins; i++){
            binnedFFT[i] = 0;
        }

        final int windowOverlap = sampleRate / 2;  // 50% overlapping windows
        final int numWindows = n / windowOverlap - 1;
        double[] windowFFT = new double[sampleRate];
        DoubleFFT_1D windowTransformer = new DoubleFFT_1D(sampleRate);
        for (int i = 0; i < numWindows; i++ ) {
            for (int j = 0; j < windowFFT.length; j++){  // slicing
                windowFFT[j] = v[i*windowOverlap+j];
            }
            HanningWindow(windowFFT, windowFFT.length);
            windowTransformer.realForward(windowFFT);  // FFT library computes coefs inplace
            final double[] windowFFTmag = getFFTmagnitude(windowFFT);  // parse FFT coefs to obtain magnitudes
            // Accumulate the magnitudes
            for (int j = 0; j < binnedFFT.length; j++){
                binnedFFT[j] += windowFFTmag[j];
            }
        }
        // Average the magnitudes. Also use logscale for convenience.
        for (int i = 0; i < binnedFFT.length; i++){
            binnedFFT[i] = Math.log(binnedFFT[i]/numWindows + 1E-8);
        }

        // don't forget to change header method immediately below !!!
        double[] output = new double[]{
            f1,
            p1,
            f33,
            p33,
            spectralEntropy
        };
        return AccStats.combineArrays(output, binnedFFT);

    }

    private static String getSanDiegoFFTHeader() {
        String header = "fmax,pmax,fmaxband,pmaxband,entropy";
        final int numBins = 10;
        for(int i=0; i<numBins; i++){
            header += ",fft" + (i+1);
        }
        return header;
    }


    /**
     * From paper:
     * A universal, accurate intensity-based classification of different physical
     * activities using raw data of accelerometer.
     * Henri Vaha-Ypya, Tommi Vasankari, Pauliina Husu, Jaana Suni and Harri Sievanen
     * https://www.ncbi.nlm.nih.gov/pubmed/24393233
     */
    private static double[] calculateMADFeatures(
            double[] x,
            double[] y,
            double[] z) {

        double[] unfilteredVM = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            if (!Double.isNaN(x[i])) {
                double vm = AccStats.getVectorMagnitude(x[i], y[i], z[i]);
                //todo should really be on vm, not vm - 1
                unfilteredVM[i] = vm - 1;
            }
        }

        // used in calculation
        int n = unfilteredVM.length;
        double N = (double) n; // avoid integer arithmetic
        double vmMean = AccStats.mean(unfilteredVM);
        double vmStd = AccStats.std(unfilteredVM, vmMean);

        // features from paper:
        // Mean amplitude deviation (MAD) describes the typical distance of data points about the mean
        double MAD = 0;
        // Mean power deviation (MPD) describes the dispersion of data points about the mean
        double MPD = 0;
        // Skewness (skewR) describes the asymmetry of dispersion of data points about the mean
        double skew = 0;
        // Kurtosis (kurtR) describes the peakedness of the distribution of data points
        double kurt = 0;
        for (int i = 0; i < n; i++) {
            double diff = unfilteredVM[i] - vmMean;
            MAD += Math.abs(diff);
            MPD += Math.pow(Math.abs(diff), 1.5);
            skew += Math.pow(diff/(vmStd + 1E-8), 3);
            kurt += Math.pow(diff/(vmStd + 1E-8), 4);
        }

        MAD /= N;
        MPD /= Math.pow(N, 1.5);
        skew *= N / ((N-1)*(N-2));
        kurt = kurt * N*(N+1)/((N-1)*(N-2)*(N-3)*(N-4)) - 3*(N-1)*(N-1)/((N-2)*(N-3));

        // don't forget to change header method immediately below !!!
        return new double[] {
            MAD,
            MPD,
            skew,
            kurt
        };
    }

    private static String getMADFeaturesHeader(){
        String header = "MAD,MPD,skew,kurt";
        return header;
    }


    private static double[] HanningWindow(double[] signal_in, int size)
    {
        for (int i = 0; i < size; i++)
        {
            signal_in[i] = (double) (signal_in[i] * 0.5 * (1.0 - Math.cos(2.0 * Math.PI * i / (size-1))));
        }
        return signal_in;
    }


    private static double[] getFFTmagnitude(double[] FFT) {
        return getFFTmagnitude(FFT, true);
    }

    //converts FFT library's complex output to only absolute magnitude
    private static double[] getFFTmagnitude(double[] FFT, boolean normalize) {
        // Get magnitudes from FFT coefficients
        double[] FFTmag = getFFTpower(FFT, normalize);
        for (int i=0; i<FFTmag.length; i++){
            FFTmag[i] = Math.sqrt(FFTmag[i]);
        }
        return FFTmag;
    }

    private static double[] getFFTpower(double[] FFT) {
        return getFFTpower(FFT, true);
    }



    private static double[] getFFTpower(double[] FFT, boolean normalize) {
        /*
         * Get powers from FFT coefficients

         * The layout of FFT is as follows (computed using JTransforms, see
         * https://github.com/wendykierp/JTransforms/blob/3c3253f240510c5f9ec700f2d9d25cfadfc857cc/src/main/java/org/jtransforms/fft/DoubleFFT_1D.java#L459):

         * If n is even then
         * FFT[2*k] = Re[k], 0<=k<n/2
         * FFT[2*k+1] = Im[k], 0<k<n/2
         * FFT[1] = Re[n/2]
         * e.g. for n=6:
         * FFT = { Re[0], Re[3], Re[1], Im[1], Re[2], Im[2] }

         * If n is odd then
         * FFT[2*k] = Re[k], 0<=k<(n+1)/2
         * FFT[2*k+1] = Im[k], 0<k<(n-1)/2
         * FFT[1] = Im[(n-1)/2]
         * e.g for n = 7:
         * FFT = { Re[0], Im[3], Re[1], Im[1], Re[2], Im[2], Re[3] }

         * See also: https://stackoverflow.com/a/5010434/3250500
        */
        final int n = FFT.length;
        final int m = (int) Math.ceil((double) n/2);
        double[] FFTpow = new double[m];
        double Re, Im;

        FFTpow[0] = FFT[0] * FFT[0];
        for (int i = 1; i < m-1; i++) {
            Re = FFT[i*2];
            Im = FFT[i*2 + 1];
            FFTpow[i] = Re * Re + Im * Im;
        }
        // The last power is a bit tricky due to the weird layout of FFT
        if (n % 2 == 0) {
            Re = FFT[n-2];  // FFT[2*m-2]
            Im = FFT[n-1];  // FFT[2*m-1]
        } else {
            Re = FFT[n-1];  // FFT[2*m-2]
            Im = FFT[1];
        }
        FFTpow[m-1] = Re * Re + Im * Im;

        if (normalize) {
            // Divide by length of the signal
            for (int i=0; i<m; i++){
                FFTpow[i] /= n*n;  // square so that after sqrt becomes 1/n
            }
        }

        return FFTpow;
    }


    /**
     * From paper:
     * Physical Activity Classification using the GENEA Wrist Worn Accelerometer
     * Shaoyan Zhang, Alex V. Rowlands, Peter Murray, Tina Hurst
     * https://www.ncbi.nlm.nih.gov/pubmed/21988935
     * See also:
     * Activity recognition using a single accelerometer placed at the wrist or ankle.
     * Mannini A, Intille SS, Rosenberger M, Sabatini AM, Haskell W.
     */
    private static double[] unileverFeatures(double[] v, int sampleRate) {
        //Compute FFT and power spectrum density
        final int n = v.length;
        final double vMean = AccStats.mean(v);

        // Initialize array to compute FFT coefs
        double[] vFFT = new double[n];
        for (int i = 0; i < n; i++)  vFFT[i] = v[i] - vMean;  // note: we remove the 0Hz freq
        HanningWindow(vFFT, vFFT.length);
        new DoubleFFT_1D(vFFT.length).realForward(vFFT);  // FFT library computes coefs inplace
        final double[] vFFTpow = getFFTpower(vFFT);  // parse FFT coefs to obtain the powers

        //Find dominant frequencies in 0.3Hz - 15Hz, also in 0.6Hz - 2.5Hz
        //Also accumulate total power in 0.3Hz - 15Hz.
        final double maxFreq = 15;
        final double minFreq = 0.3;
        final double FFTinterval = sampleRate / (1.0 * n); // (Hz)
        double f1=-1, f2=-1, f625=-1;
        double p1=0, p2=0, p625=0;
        double totalPower = 0.0;
        for (int i = 0; i < vFFTpow.length; i++) {
            double freq = FFTinterval * i;
            if (freq < minFreq || freq > maxFreq) continue;
            double p = vFFTpow[i];
            totalPower += p;
            if (p > p1) {
                f2 = f1;
                p2 = p1;
                f1 = freq;
                p1 = p;
            } else if (p > p2) {
                f2 = freq;
                p2 = p;
            }
            if (p > p625 && freq > 0.6 && freq < 2.5) {
                f625 = freq;
                p625 = p;
            }
        }
        // Use logscale for convenience
        totalPower = Math.log(totalPower + 1E-8);
        p1 = Math.log(p1 + 1E-8);
        p2 = Math.log(p2 + 1E-8);
        p625 = Math.log(p625 + 1E-8);

        // don't forget to change header method immediately below !!!
        return new double[] {
            f1,
            p1,
            f2,
            p2,
            f625,
            p625,
            totalPower
        };
    }

    private static String getUnileverFeaturesHeader(){
        String header = "f1,p1,f2,p2,f625,p625,totalPower";
        return header;
    }


    /**
     * From paper:
     * Estimating sleep parameters using an accelerometer without sleep diary
     * van Hees et al. 2018
     * https://www.nature.com/articles/s41598-018-31266-z
     * See also:
     * A novel, open access method to assess sleep duration using a wrist-worn accelerometer
     * van Hees et al. 2015
     *
     * avgArmAngel is only defined for epoch_length >= 5 sec and avgArmAngelAbsDiff
     * is only defined for epoch_length >= 10 sec. If epoch_length is not valid, we will return 0.
     * For more details, please refer to the references above.
     */
    private static double[] calculateArmFeatures(double[] x,
                                                 double[] y,
                                                 double[] z,
                                                 int sampleRate) {
        int window_len = 5; // 5-sec
        if (x.length / sampleRate < window_len) {
            return new double[] {
                    0, 0 // if less than 5sec, just return zeros
            };
        } else {
            // 1. 5-sec rolling medians
            int k = window_len * sampleRate;
            double[] rollingMedianX = medianSlidingWindow(x, k);
            double[] rollingMedianY = medianSlidingWindow(y, k);
            double[] rollingMedianZ = medianSlidingWindow(z, k);

            // 2. compute arm angel
            double[] angelZ = new double[rollingMedianX.length];

            for (int i = 0; i < rollingMedianX.length; i++) {
                double tmp = rollingMedianZ[i] / (Math.pow(rollingMedianX[i], 2) + Math.pow(rollingMedianY[i], 2));
                angelZ[i] = Math.atan(tmp) * 180 / Math.PI;
            }

            // 3. consecutive 5-sec avg
            double[] fiveSecAvg = computeFiveSecAvg(angelZ, sampleRate);
            double avgArmAngel = AccStats.mean(fiveSecAvg);

            // 4. Absolute difference between successive values
            //    ill-defined when we have fewer 10 sec for the epoch length
            double avgArmAngelAbsDiff = 0;
            if (x.length/sampleRate >= 10) {
                double[] absoluteAvgDiff = computeAbsoluteDiff(fiveSecAvg);
                avgArmAngelAbsDiff = AccStats.mean(absoluteAvgDiff);
            }

            // don't forget to change header method immediately below !!!
            return new double[]{
                    avgArmAngel,
                    avgArmAngelAbsDiff
            };
        }
    }

    private static String getArmFeaturesHeader(){
        String header = "avgArmAngel,avgArmAngelAbsDiff";
        return header;
    }

    /**
     * Obtain the rolling window median of window of size k
     */
    public static double[] medianSlidingWindow(double[] nums, int k) {
        double[] res=new double[nums.length-k+1];
        List<Double> list = new ArrayList<Double>();
        for(int i=0;i<k;i++){
            list.add(nums[i]);
        }
        Collections.sort(list);
        res[0]=(k%2==0)?((double)list.get(k/2-1)+(double)list.get(k/2))/2:list.get(k/2);
        for(int i=0;i<nums.length-k;i++){
            double left=nums[i];
            double right=nums[i+k];
            int index=Collections.binarySearch(list,right);
            if(index>=0) list.add(index,right);
            else list.add(-index-1,right);
            index=Collections.binarySearch(list,left);
            list.remove(index);
            res[i+1]=(k%2==0)?((double)list.get(k/2-1)+(double)list.get(k/2))/2:list.get(k/2);
        }
        return res;
    }

    private static double[] computeFiveSecAvg(double[] x, int sampleRate) {
        int avgsLen = (int)Math.ceil(x.length/(5.0*sampleRate));
        double[] avgs = new double[avgsLen];
        int count = 0;
        double sum = 0;
        int j = 0;
        for (int i = 0; i < x.length; i++) {
            count++;
            sum += x[i];
            if (count==5*sampleRate || i==x.length-1) {
                avgs[j] = sum/count;
                j++;
                sum = 0;
                count = 0;
            }
        }
        return avgs;
    }

    private static double[] computeAbsoluteDiff(double[] x) {
        double[] res = new double[x.length-1];
        for (int i = 1; i < x.length; i++) {
            res[i-1] = Math.abs(x[i]-x[i-1]);
        }
        return res;
    }


    /**
     * This assumes the correlation is zero if one of the axes is all constant.
     * In reality, correlation is undefined in such cases, but for our
     * particular case we shall assume that the axes are independent at rest
     */
    private static double correlation(double[] vals1, double[] vals2, int lag) {
        double res = AccStats.correlation(vals1, vals2, lag);
        if (!Double.isFinite(res)) return 0.0;
        return res;
    }


    private static double correlation(double[] vals1, double[] vals2) {
        return correlation(vals1, vals2, 0);
    }


}
