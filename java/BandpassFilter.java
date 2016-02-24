//BSD 2-Clause, (c) 2014: A.Doherty (Oxford), D.Jackson, N.Hammerla (Newcastle)
import java.util.List;

// (This inner class should probably be in another file)
// [dgj] Butterworth 4th-order bandpass filter
public class BandpassFilter {

	public final static int BUTTERWORTH4_ORDER = 4;
	public final static int BUTTERWORTH4_NUM_COEFFICIENTS = (BUTTERWORTH4_ORDER * 2 + 1);

    // Filter coefficients
    private double B[];
    private double A[];
    
	// Final/initial conditions
    private double z[];
	
	// Calculate coefficients for a 4th order Butterworth bandpass filter.
	// Based on http://www.exstrom.com/journal/sigproc/
	// Copyright (C) 2014 Exstrom Laboratories LLC
	void CoefficientsButterworth4BP(double W1, double W2, double B[], double A[])
	{
		int i, j;		
		// Calculate B coefficients as if for a Butterworth lowpass filter. 
		int prev = BUTTERWORTH4_ORDER;
		int tcof[] = new int[BUTTERWORTH4_ORDER + 1];
		tcof[0] = 1;
		tcof[1] = BUTTERWORTH4_ORDER;
		for (i = 2; i <= (BUTTERWORTH4_ORDER / 2); i++)
		{
			prev = (BUTTERWORTH4_ORDER - i + 1) * prev / i;
			tcof[i] = prev;
			tcof[BUTTERWORTH4_ORDER - i] = prev;
		}
		tcof[BUTTERWORTH4_ORDER - 1] = BUTTERWORTH4_ORDER;
		tcof[BUTTERWORTH4_ORDER] = 1;

		// Calculate the scaling factor for the B coefficients of Butterworth
		// bandpass filter (so the filter response has a maximum value of 1).
		double ctt = 1.0 / Math.tan(Math.PI * (W2 - W1) / 2.0);
		double sfr = 1.0;
		double sfi = 0.0;
		for (i = 0; i < BUTTERWORTH4_ORDER; i++)
		{
			double parg = Math.PI * (double)(2*i + 1) / (double)(2*BUTTERWORTH4_ORDER);
			double a = (sfr + sfi) * ((ctt + Math.sin(parg)) - Math.cos(parg));
			double b = sfr * (ctt + Math.sin(parg));
			double c = -sfi * Math.cos(parg);
			sfr = b - c;
			sfi = a - b - c;
		}
		double sf_bwbp = (1.0 / sfr);

		// Update coefficients to Butterworth bandpass filter, and apply scaling factor
		for (i = 0; i < BUTTERWORTH4_ORDER; ++i)
		{
			double sign = ((i & 1) != 0) ? -1 : 1;
			B[2 * i] = sign * sf_bwbp * tcof[i];
			B[2 * i + 1] = 0;
		}
		B[2 * BUTTERWORTH4_ORDER] = sf_bwbp * tcof[BUTTERWORTH4_ORDER];


		// Begin to calculate the A coefficients
		double cp = Math.cos(Math.PI * (W2 + W1) / 2.0);
		double theta = Math.PI * (W2 - W1) / 2.0;
		double s2t = 2.0 * Math.sin(theta) * Math.cos(theta);
		double c2t = 2.0 * Math.cos(theta) * Math.cos(theta) - 1.0;

		// Trinomials
		double c[] = new double[BUTTERWORTH4_ORDER * 2];
		double b[] = new double[BUTTERWORTH4_ORDER * 2];
		for (i = 0; i < BUTTERWORTH4_ORDER; i++)
		{
			double parg = Math.PI * (double)(2*i + 1) / (double)(2*BUTTERWORTH4_ORDER);
			double z = 1.0 + s2t * Math.sin(parg);
			c[2 * i] = c2t / z;
			c[2 * i + 1] = s2t * Math.cos(parg) / z;
			b[2 * i] = -2.0*cp * (Math.cos(theta) + Math.sin(theta) * Math.sin(parg)) / z;
			b[2 * i + 1] = -2.0*cp * Math.sin(theta) * Math.cos(parg) / z;
		}

		// Multiply trinomials together and returns the coefficients of the resulting polynomial.
		double a[] = new double[4 * BUTTERWORTH4_ORDER];
		{
			a[2] = c[0]; a[3] = c[1]; a[0] = b[0]; a[1] = b[1];
			for (i = 1; i < BUTTERWORTH4_ORDER; i++)
			{
				a[2 * (2*i + 1)] += c[2*i] * a[2 * (2*i - 1)] - c[2*i + 1] * a[2 * (2*i - 1) + 1];
				a[2 * (2*i + 1) + 1] += c[2*i] * a[2 * (2*i - 1) + 1] + c[2*i + 1] * a[2 * (2*i - 1)];
				for (j = 2 * i; j > 1; j--)
				{
					a[2*j] += b[2*i] * a[2 * (j-1)] - b[2*i + 1] * a[2 * (j-1) + 1] + c[2*i] * a[2 * (j-2)] - c[2*i + 1] * a[2 * (j-2) + 1];
					a[2*j + 1] += b[2*i] * a[2 * (j-1) + 1] + b[2*i + 1] * a[2* (j-1)] + c[2*i] * a[2 * (j-2) + 1] + c[2*i + 1] * a[2 * (j-2)];
				}
				a[2] += b[2*i] * a[0] - b[2*i + 1] * a[1] + c[2*i];
				a[3] += b[2*i] * a[1] + b[2*i + 1] * a[0] + c[2*i + 1];
				a[0] += b[2*i];
				a[1] += b[2*i + 1];
			}
		}

		// Read out results as A coefficients
		A[1] = a[0];
		A[0] = 1.0;
		A[2] = a[2];
		for (i = 3; i <= 2 * BUTTERWORTH4_ORDER; i++)
		{
			A[i] = a[2*i - 2];
		}

		return;
	}
	
	// Constructs 4th order Butterworth bandpass filter between Fc1 and Fc2 at rate Fs.
	public BandpassFilter(double Fc1, double Fc2, double Fs)
	{
		// Calculate normalised cut-offs
		double W1 = Fc1 / (Fs / 2);
		double W2 = Fc2 / (Fs / 2);
	
		// Create coefficients
		B = new double[BUTTERWORTH4_NUM_COEFFICIENTS];
		A = new double[BUTTERWORTH4_NUM_COEFFICIENTS];
		
		// Calculate coefficients
		CoefficientsButterworth4BP(W1, W2, B, A);
		
		// [debug] Dump coefficients
        Boolean debug = false;
		if (debug) {
			System.out.print("B = [ ");
			for (int i = 0; i < BUTTERWORTH4_NUM_COEFFICIENTS; i++) {
			    System.out.print("" + B[i] + " ");
			}
			System.out.print("]\n");
			System.out.print("A = [ ");
			for (int i = 0; i < BUTTERWORTH4_NUM_COEFFICIENTS; i++) {
			    System.out.print("" + A[i] + " ");
			}
			System.out.print("]\n");
		}
		
		// Create final/initial condition tracker
		z = new double[BUTTERWORTH4_NUM_COEFFICIENTS];
		reset();
	}
	
	// Reset state tracking
	public void reset() {
		for (int i = 0; i < z.length; i++) { 
			z[i] = 0; 
		}
	}
	
	// Apply the filter to the specified data
	public void filter(double X[], int offset, int count) {
		int m, i;
		
		z[BUTTERWORTH4_NUM_COEFFICIENTS - 1] = 0;
		for (m = offset; m < offset + count; m++) {
			double oldXm = X[m];
			double newXm = B[0] * oldXm + z[0];
			for (i = 1; i < BUTTERWORTH4_NUM_COEFFICIENTS; i++) {
				z[i - 1] = B[i] * oldXm + z[i] - A[i] * newXm;
			}
			X[m] = newXm;
		}
		return;
	}
	
	// Apply the filter to the specified data
	public void filter(double X[]) {
		filter(X, 0, X.length);
	}		
	
	// Apply the filter to the specified data
	public void filter(List<Double> X, int offset, int count) {
		int m, i;
		
		z[BUTTERWORTH4_NUM_COEFFICIENTS - 1] = 0;
		for (m = offset; m < offset + count; m++) {
			double oldXm = X.get(m);
			double newXm = B[0] * oldXm + z[0];
			for (i = 1; i < BUTTERWORTH4_NUM_COEFFICIENTS; i++) {
				z[i - 1] = B[i] * oldXm + z[i] - A[i] * newXm;
			}
			X.set(m, newXm);
		}
		return;
	}
	
	// Apply the filter to the specified data
	public void filter(List<Double> X) {
		filter(X, 0, X.size());
		return;
	}
	
}
