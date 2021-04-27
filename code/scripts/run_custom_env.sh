
../build/whiskit_gui \
--PRINT 2 \
--CDIST 50 \
--CPITCH -0 \
--CYAW 180 \
--BLOW 1 \
--DEBUG 0 \
--OBJECT 4 \
--file_env "../data/environment/env_6.obj" \
--ACTIVE 1 \
--TIME_STOP 1.0 \
## WHISKER SETTINGS
# --WHISKER_NAMES R \
--ORIENTATION 90 -90 0 \
--POSITION -95 0 -140 \
# --POSITION 0 0 0 \
--SAVE_VIDEO 1 \
--SAVE 0 \
--ObjX 0.0 \
--ObjY 0.0 \
--ObjZ 0.0 \
## OBJECT OREIENTATIONS (4D Quaternion) *make sure to use normalized values
--ObjQx 0.0 \
--ObjQy 0.0 \
--ObjQz 0.0 \
--ObjQw 1.0 \
## OBJECT OREIENTATIONS (3D Quaternion)
--ObjYAW 0.0 \
--ObjPITCH 0.0 \
--ObjROLL 0.0 \
--file_video "../output/video_object.mp4" \
--dir_out ../output/full_array_env_active