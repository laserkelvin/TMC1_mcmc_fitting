#!/bin/bash

GOT_PATH=Projects/GOTHAM/data

rclone sync dropbox:$GOT_PATH/data ./raw

rclone sync dropbox:$GOT_PATH/phase1_SI/supplementary ./supplementary

rclone sync dropbox:$GOT_PATH/gotham_catalogs_trimmed ./gotham_catalogs

