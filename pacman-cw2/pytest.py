num = [i for i in range(0,1001)]
import os
for i in num: 
	os.system('python pacman.py -p QLearnAgent -l smallGrid  -a alpha=0.2 -x '+str(i)+' -n '+str(i+10)+' -q')

