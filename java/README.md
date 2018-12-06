biobankAccelerometerAnalysis-new-features: java
======================

Requirements: java 1.8

#### Compilation

Run the scripts in outer folder to recompile, for Windows: `compile_java.bat`, for linux/OSX `compile_java.sh`.

Alternatively, run this command (in this directory):
``` bash
javac -cp JTransforms-3.1-with-dependencies.jar *.java
```

This is because we are using the JTransforms FFT library, so java needs to know where it is.

#### Eclipse

If using eclipse (a recommended Java IDE), you should right click the `JTransforms-3.1-with-dependencies.jar` and click 'add to build path' and eclipse will then be able to compile and run it.
