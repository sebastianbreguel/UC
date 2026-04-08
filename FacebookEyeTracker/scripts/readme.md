Here we have all the files to run the eyetracker, process the data, match it with the json data and generate the visualizations.

- gazeProcess.py: once the eye tracking data is saved into a csv file, this clean the data.
- generate.py: run the eye_tracker and track the data
- match.py: from the screenshots and csv processed, make the visualization, Heat map and Scanpath
- utils.py: other functions
- visualization: run the gaze and scanpath plots.

folder Visualizations:

- gazeheatplot.py: from a gaze.csv data and a image base, generate the heatmap plot
- scanpathPlot.py: generate the scanplot from a gaze.csv and a image
