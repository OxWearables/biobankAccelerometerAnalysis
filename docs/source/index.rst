.. accelerometer documentation master file, created by
   sphinx-quickstart on Tue Nov 27 12:48:46 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: accelerometerLogo.png

A tool to extract meaningful health information from large accelerometer
datasets. The software generates time-series and summary metrics useful for
answering key questions such as how much time is spent in sleep, sedentary
behaviour, or doing physical activity.

************
Installation
************

*Minimum requirements*: Python>=3.7, Java 8 (1.8)

The following instructions make use of Anaconda to meet the minimum requirements:

#. Download & install `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__ (light-weight version of Anaconda).
#. (Windows) Once installed, launch the **Anaconda Prompt**.
#. Create a virtual environment::

     $ conda create -n accelerometer python=3.9 openjdk pip

   This creates a virtual environment called :code:`accelerometer` with Python version 3.9, OpenJDK, and Pip.
#. Activate the environment::

     $ conda activate accelerometer

   You should now see ``(accelerometer)`` written in front of your prompt.
#. Install :code:`accelerometer`::

     $ pip install accelerometer

You are all set! The next time that you want to use :code:`accelerometer`, open the Anaconda Prompt and activate the environment (step 4). If you see ``(accelerometer)`` in front of your prompt, you are ready to go!

***************
Getting started
***************
To extract a summary of movement (average sample vector magnitude) and
(non)wear time from raw Axivity .CWA (or gzipped .cwa.gz) accelerometer files:

.. code-block:: console

    $ accProcess data/sample.cwa.gz
    <output written to data/sample-outputSummary.json>
    <time series output written to data/sample-timeSeries.csv.gz>

The main output JSON will look like:

.. code-block:: console

    {
        file-name: "sample.cwa.gz",
        file-startTime: "2014-05-07 13:29:50",
        file-endTime: "2014-05-13 09:49:50",
        acc-overall-avg(mg): 32.78149,
        wearTime-overall(days): 5.8,
        nonWearTime-overall(days): 0.04,
        quality-goodWearTime: 1
    }

To visualise the time series and activity classification output:

.. code-block:: console

    $ accPlot data/sample-timeSeries.csv.gz
    <output plot written to data/sample-plot.png>

.. figure:: samplePlot.png

    Output plot of overall activity and class predictions for each 30sec time window

.. The underlying modules can also be called in custom python scripts:

.. .. code-block:: python

..     from accelerometer import summariseEpoch
..     summary = {}
..     epochData, labels = summariseEpoch.getActivitySummary("sample-epoch.csv.gz",
..             "sample-nonWear.csv.gz", summary)
..     # <nonWear file written to "sample-nonWear.csv.gz" and dict "summary" updated
..     # with outcomes>


***************
Citing our work
***************
When using this tool, please consider the works listed in `CITATION.md <https://github.com/OxWearables/biobankAccelerometerAnalysis/blob/master/CITATION.md>`_.


*******
Licence
*******
See `LICENSE.md <https://github.com/OxWearables/biobankAccelerometerAnalysis/blob/master/LICENSE.md>`_.


************
Acknowledgements
************
We would like to thank all our code contributors and manuscript co-authors.
`Contributors Graph <https://github.com/OxWearables/biobankAccelerometerAnalysis/graphs/contributors>`_.



.. toctree::
   :maxdepth: 1
   :caption: Contents:

   usage
   methods
   cliapi
   datadict


******************
Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
