import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;

class Resample{

    //inspired by http://www.java2s.com/Code/Java/Collections-Data-Structure/LinearInterpolation.htm
    //in small tests this method has matched python scipy.interpolate.interp1d
    //http://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
    public static final void interpLinear(
            List<Long> time, //time in milliseconds
            List<Double> x,
            List<Double> y,
            List<Double> z,
            long[] timeI, //time in milliseconds
            double[] xNew,
            double[] yNew,
            double[] zNew) throws IllegalArgumentException {
        if (time.size() != x.size()) {
            throw new IllegalArgumentException("time and x must be the same length");
        }
        if (time.size() == 1) {
            throw new IllegalArgumentException("time must contain more than one value");
        }
        double[] dtime = new double[time.size() - 1];
        double[] dx = new double[time.size() - 1];
        double[] dy = new double[time.size() - 1];
        double[] dz = new double[time.size() - 1];
        double[] xSlope = new double[time.size() - 1];
        double[] ySlope = new double[time.size() - 1];
        double[] zSlope = new double[time.size() - 1];
        double[] xIntercept = new double[time.size() - 1];
        double[] yIntercept = new double[time.size() - 1];
        double[] zIntercept = new double[time.size() - 1];

        // Calculate the line equation (i.e. slope and intercept) between each point
        for (int i = time.size()-2; i >= 0; i--) {
            dtime[i] = time.get(i + 1) - time.get(i);
            if (dtime[i] <= 0){
                time.set(i, time.get(i+1) - 1);
                dtime[i] = 1;
            }
            dx[i] = x.get(i + 1) - x.get(i);
            dy[i] = y.get(i + 1) - y.get(i);
            dz[i] = z.get(i + 1) - z.get(i);
            xSlope[i] = dx[i] / dtime[i];
            ySlope[i] = dy[i] / dtime[i];
            zSlope[i] = dz[i] / dtime[i];
            xIntercept[i] = x.get(i) - time.get(i) * xSlope[i];
            yIntercept[i] = y.get(i) - time.get(i) * ySlope[i];
            zIntercept[i] = z.get(i) - time.get(i) * zSlope[i];
        }
        
        // Perform the interpolation here
        for (int i = 0; i < timeI.length; i++) {
            if ((timeI[i] > time.get(time.size() - 1)) || (timeI[i] < time.get(0))) {
                xNew[i] = Double.NaN;
                yNew[i] = Double.NaN;
                zNew[i] = Double.NaN;
            }
            else {
                int loc = Collections.binarySearch(time, timeI[i]);
                if (loc < -1) {
                    loc = -loc - 2;
                    double xNewInstant = xSlope[loc] * timeI[i] + xIntercept[loc];
                    double yNewInstant = ySlope[loc] * timeI[i] + yIntercept[loc];
                    double zNewInstant = zSlope[loc] * timeI[i] + zIntercept[loc];
                    xNew[i] = xNewInstant; 
                    yNew[i] = yNewInstant;
                    zNew[i] = zNewInstant;
                }
                else {
                    xNew[i] = x.get(loc);
                    yNew[i] = y.get(loc);
                    zNew[i] = z.get(loc);
                }
            }
        }
    }

}
