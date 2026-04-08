import argparse
import os


# run generate.py


argpase = argparse.ArgumentParser()
## args: duration, name, width, height, input_file, output_file
argpase.add_argument("duration", type=int, help="total seconds to collect data")
argpase.add_argument("name", type=str, help="name of the output file")
argpase.add_argument("width", type=int, help="Screen width", default=1920)
argpase.add_argument("height", type=int, help="Screen height", default=1080)

args = vars(argpase.parse_args())
duration = args["duration"]
name = args["name"]
width = args["width"]
height = args["height"]
base = f"images/{name}.png"

# input for the processing
input_file = f"gaze/my_gaze_data_{name}"

print("Runnign eye tracker")
os.system(f"python generate.py {duration} {name}")

print("Processing gaze data")
os.system(f"python gazeProcess.py {input_file}.csv {input_file}_clean.csv {width} {height}")

print("Generating heatmap")
os.system(f"python gazeheatplot.py {input_file}_clean.csv {width} {height} -b {base} -n 125    -sd 15 -o heatmap/{name}.png")
