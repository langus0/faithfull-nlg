
path=$1

rclone copy results2/${path} results/${path} --progress --multi-thread-streams=4
