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

## How to contribute
We shall stick to a test-driven development process as it is the only way we can make sure that
our code is robust. Whenever new code is being written, one shall also write the respective tests
and add them to the Junit test cases. The unit tests will be run when a pull request
is created.

### How to run tests
1. Download junit platform console stand-alone [jar](https://repo1.maven.org/maven2/org/junit/platform/junit-platform-console-standalone/1.6.0/junit-platform-console-standalone-1.6.0.jar) to `java` folder.
2. Compile tests
3. Run the tests

```bash
javac -cp JTransforms-3.1-with-dependencies.jar *.java
javac -d out -cp out:junit-platform-console-standalone-1.6.0.jar:.:Tests Tests/*.java
java -jar junit-platform-console-standalone-1.6.0.jar --classpath out:. --scan-class-path
```
