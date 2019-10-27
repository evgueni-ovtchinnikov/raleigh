@ECHO OFF
SETLOCAL
SET args=
SET count=0
:Loop
IF "%1"=="" GOTO Continue
rem ECHO %1
IF %count%==0 SET script=%1.py
IF %count%==1 SET args=%1
IF %count% GTR 1 SET args=%args% %1
SET /A count=count+1
SHIFT
GOTO Loop
:Continue
COPY .\tests\%script% .
rem ECHO %script% %args%
python %script% %args%
DEL %script%