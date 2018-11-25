rm output.mp4
rm output_audio.mp4

ffmpeg -framerate 4 -i generateframes/test/test%d.jpg -c:v libx264 -pix_fmt yuv420p -crf 23 output.mp4

ffmpeg -i output.mp4 -i projects/jee/jee.mp3 -codec copy -shortest output_audio.mp4
