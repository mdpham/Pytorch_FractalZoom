frames:
	cd generateframes/pytorch && python mandelbrotzoom.py

video:
	bash video.sh

test:
	make frames
	make video