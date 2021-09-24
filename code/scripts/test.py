import subprocess
str1 = "../build/whiskit_gui \
--PRINT 2 \
--CDIST 50 \
--WHISKER_NAMES R \
--TIME_STOP 1.0 \
--CPITCH 0 \
--CYAW 180 \
--BLOW 1  \
--OBJECT 5 \
--ACTIVE 1 \
--SAVE_VIDEO 0 \
--SAVE 1 "

s = subprocess.getoutput([str1])