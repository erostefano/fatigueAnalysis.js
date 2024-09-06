#!/bin/sh
dir="yawning"
ffmpeg -i ./$dir/$dir.mp4 -vf "fps=60" ./$dir/frames/frame_%04d.png
