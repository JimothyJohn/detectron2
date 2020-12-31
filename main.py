from projectron.Projectron import Projectron
import argparse
tool = Projectron('keypoint')
tool.RecordImage('/scratch/andrew.jpg','/scratch/andrewmask.jpg')

# Get input file, output target, and inference type
parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Input file', dest='input', required=True)
parser.add_argument('--stream', help='Mask file', type=bool, dest='stream', default=False)
parser.add_argument('--record', help='Output file', dest='output')
parser.add_argument('--inference', help='Type of inference requested (detection, segmentation, keypoint, panoptic)', dest='inference', default='pointrend')
parser.add_argument('--start', help='Start of video', type=int, dest='start', default=0)
parser.add_argument('--length', help='Length of video', type=int, dest='length', default=256)
args = parser.parse_args()