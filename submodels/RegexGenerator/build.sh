#! /bin/sh

if ! command -v ant 2>&1 >/dev/null && ! [ -d 'apache-ant-1.10.11' ]; then
wget https://dlcdn.apache.org//ant/binaries/apache-ant-1.10.11-bin.zip
unzip apache-ant-1.10.11-bin.zip
rm apache-ant-1.10.11-bin.zip
fi
export PATH=$PATH:`pwd`/apache-ant-1.10.11/bin

cd origin_src/ConsoleRegexTurtle
ant -Dplatforms.JDK_1.7.home=$JAVA_HOME

cd ../..
cp origin_src/ConsoleRegexTurtle/dist/ConsoleRegexTurtle.jar .
