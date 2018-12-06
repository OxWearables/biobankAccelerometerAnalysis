//BSD 2-Clause, (c) 2014: A.Doherty (Oxford), D.Jackson, N.Hammerla (Newcastle)
import java.util.Arrays;
import java.util.List;

// (This inner class should probably be in another file)
// [dgj] Butterworth 4th-order lowpass filter
public class LowpassFilter {

	public final static int BUTTERWORTH4_ORDER = 4;
	public final static int BUTTERWORTH4_NUM_COEFFICIENTS = (BUTTERWORTH4_ORDER + 1);

    // Filter coefficients
    private double B[];
    private double A[];
    
	// Final/initial conditions
    private double z[];
	
	// Calculate coefficients for a 4th order Butterworth lowpass filter.
	// Based on http://www.exstrom.com/journal/sigproc/
	// Copyright (C) 2014 Exstrom Laboratories LLC
	void CoefficientsButterworth4LP(double W, double B[], double A[])
	{
		// (Bit hacky:) treat a negative value as a high-pass
		Boolean highpass = false;
		if (W < 0) { W = -W; highpass = true; }
	
		int i, j;		
		
		// Calculate B coefficients for a Butterworth lowpass/highpass filter. 
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
		// lowpass filter (so the filter response has a maximum value of 1).
		double fcf = W;
		double omega = Math.PI * fcf;
		double fomega = Math.sin(omega);
		double parg0 = Math.PI / (double)(2 * BUTTERWORTH4_ORDER);
		double sf = 1.0;
		for (i = 0; i < BUTTERWORTH4_ORDER / 2; ++i)
		{
			sf *= 1.0 + fomega * Math.sin((double)(2 * i + 1) * parg0);
		}
		
		if (highpass) {
			fomega = Math.cos(omega / 2.0);											// High-pass
			if (BUTTERWORTH4_ORDER % 2 != 0) { sf *= fomega + Math.sin(omega / 2.0); }  // Odd order high-pass
		} else {
			fomega = Math.sin(omega / 2.0);											// Low-pass
			if (BUTTERWORTH4_ORDER % 2 != 0) { sf *= fomega + Math.cos(omega / 2.0); }	// Odd order low-pass
		}

		// Final scaling factor
		sf = Math.pow(fomega, BUTTERWORTH4_ORDER) / sf;		
		
		// Update the coefficients by applying the scaling factor
		for (i = 0; i < BUTTERWORTH4_ORDER; ++i) {
			B[i] = sf * tcof[i];
		}
		B[BUTTERWORTH4_ORDER] = sf * tcof[BUTTERWORTH4_ORDER];		
		
		if (highpass) {
			for (i = 1; i <= BUTTERWORTH4_ORDER; i += 2) { B[i] = -B[i]; }
		}
		
		// Begin to calculate the A coefficients for a high-pass or low-pass Butterworth filter
		double theta = Math.PI * W;
		
		// Binomials
		double b[] = new double[2 * BUTTERWORTH4_ORDER];
		for (i = 0; i < BUTTERWORTH4_ORDER; i++)
		{
			double parg = Math.PI * (double)(2*i + 1) / (double)(2*BUTTERWORTH4_ORDER);
			double a = 1.0 + Math.sin(theta) * Math.sin(parg);
			b[2 * i] = -Math.cos(theta) / a;
			b[2 * i + 1] = -Math.sin(theta) * Math.cos(parg) / a;
		}

		// Multiply binomials together and returns the coefficients of the resulting polynomial.
		double a[] = new double[2 * BUTTERWORTH4_ORDER];
		for (i = 0; i < BUTTERWORTH4_ORDER; i++)
		{
			for (j = i; j > 0; --j)
			{
				a[2 * j] += b[2 * i] * a[2 * (j - 1)] - b[2 * i + 1] * a[2 * (j - 1) + 1];
				a[2 * j + 1] += b[2 * i] * a[2 * (j - 1) + 1] + b[2 * i + 1] * a[2 * (j - 1)];
			}
			a[0] += b[2 * i];
			a[1] += b[2 * i + 1];
		}

		// Read out results as A coefficients for high-pass or low-pass filter.
		A[1] = a[0];
		A[0] = 1.0;
		A[2] = a[2];
		for (i = 3; i <= BUTTERWORTH4_ORDER; ++i)
		{
			A[i] = a[2 * i - 2];
		}

		return;
	}
	
	// Constructs 4th order Butterworth lowpass filter with cutoff Fc at rate Fs.
	public LowpassFilter(double Fc, double Fs)
	{
		// Calculate normalised cut-off
		double W = Fc / (Fs / 2);

		// Create coefficients
		B = new double[BUTTERWORTH4_NUM_COEFFICIENTS];
		A = new double[BUTTERWORTH4_NUM_COEFFICIENTS];
		
		// Calculate coefficients
		CoefficientsButterworth4LP(W, B, A);
		
		// [debug] Dump coefficients
        Boolean debug = false;
		if (debug) {
			System.out.println("B = " + Arrays.toString(B));
			System.out.println("A = " + Arrays.toString(A));
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
	public void filter(double[] X, int offset, int count) {
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
	}
	
	// Additionally, returns the filtered-out signal
	public double[] filterWithRemainder(double[] X, int offset, int count) {
		double[] remainder = X.clone();
		filter(X, offset, count);
		for (int m = 0; m < X.length; m++) {
			if (m<offset || m>=offset+count) {
				remainder[m] = 0;
			} else {
				remainder[m] = X[m] - remainder[m];
			}
		}
		return remainder;
	}
	
	// Apply the filter to the specified data
	public void filter(double[] X) {
		filter(X, 0, X.length);
	}		

	
}
