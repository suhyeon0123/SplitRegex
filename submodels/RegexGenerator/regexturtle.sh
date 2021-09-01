#!/bin/sh
#Executes the command-line version of RegextTurtle; automatically sets the JAVA VM memory size based on the available system memory
MEMSYSTEM=$(free -m | grep Mem: | tr -s ' ' | cut -d' ' -f2)
MAXMEM=$(( MEMSYSTEM-512 ))
XMSMEM=$(( MAXMEM/2 ))
echo "System memory:"$MEMSYSTEM "Mbytes"
echo "RegexTurtle is going to use this amount of the system memory:"$MAXMEM "Mbytes"
exec java -Xmx${MAXMEM}M -Xms${XMSMEM}M -jar "origin_src/ConsoleRegexTurtle/dist/ConsoleRegexTurtle.jar" $@
