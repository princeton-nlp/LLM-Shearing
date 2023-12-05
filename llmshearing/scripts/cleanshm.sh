#!/bin/bash
for file in /dev/shm/*_locals; do
   if [ -e "$file" ]; then
       echo "Removing $file..."
       rm "$file"
   fi
done
