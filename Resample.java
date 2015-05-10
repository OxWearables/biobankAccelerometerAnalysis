import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;

class Resample{

    //inspired by http://www.java2s.com/Code/Java/Collections-Data-Structure/LinearInterpolation.htm
    public static final void interpLinear(
            List<Long> time,
            List<Double> x,
            List<Double> y,
            List<Double> z,
            long[] timeI,
            double[] xi,
            double[] yi,
            double[] zi) throws IllegalArgumentException {
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
            //should the line below not be PLUS???
            xIntercept[i] = x.get(i) - time.get(i) * xSlope[i];
            yIntercept[i] = y.get(i) - time.get(i) * ySlope[i];
            zIntercept[i] = z.get(i) - time.get(i) * zSlope[i];
        }
        // Perform the interpolation here
        for (int i = 0; i < timeI.length; i++) {
            if ((timeI[i] > time.get(time.size() - 1)) || (timeI[i] < time.get(0))) {
                xi[i] = Double.NaN;
                yi[i] = Double.NaN;
                zi[i] = Double.NaN;
            }
            else {
                int loc = Collections.binarySearch(time, timeI[i]);
                if (loc < -1) {
                    loc = -loc - 2;
                    xi[i] = xSlope[loc] * timeI[i] + xIntercept[loc];
                    yi[i] = ySlope[loc] * timeI[i] + yIntercept[loc];
                    zi[i] = zSlope[loc] * timeI[i] + zIntercept[loc];
                }
                else {
                    xi[i] = x.get(loc);
                    yi[i] = y.get(loc);
                    zi[i] = z.get(loc);
                }
            }
        }
    }

}
