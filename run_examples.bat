@ECHO OFF
SETLOCAL
SET args=
SET count=0
:Loop
IF "%1"=="" GOTO Continue
rem ECHO %1
IF %count%==0 SET folder=%1
IF %count%==1 SET script=%1.py
IF %count%==2 SET args=%1
IF %count% GTR 2 SET args=%args% %1
SET /A count=count+1
SHIFT
GOTO Loop
:Continue
COPY .\raleigh\examples\%folder%\%script% .
rem ECHO %script%
python %script% %args%
DEL %script%