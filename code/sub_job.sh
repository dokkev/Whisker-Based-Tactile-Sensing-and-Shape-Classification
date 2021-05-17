for i in 0 1; do
    startID=$((i*100+1));
    export STARTID=$startID
    echo "ID: $startID";
    sleep 1
    msub -v STARTID scheduler.sh
done
