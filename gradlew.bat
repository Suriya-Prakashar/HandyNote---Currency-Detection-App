@echo off
setlocal

set APP_HOME=%~dp0
set WRAPPER_JAR=%APP_HOME%gradle\wrapper\gradle-wrapper.jar
set WRAPPER_SHARED_JAR=%APP_HOME%gradle\wrapper\gradle-wrapper-shared.jar
set GRADLE_CLI_JAR=%APP_HOME%gradle\wrapper\gradle-cli-8.2.jar
set GRADLE_LAUNCHER_JAR=%APP_HOME%gradle\wrapper\gradle-launcher-8.2.jar
set CLASSPATH=%WRAPPER_JAR%;%WRAPPER_SHARED_JAR%;%GRADLE_CLI_JAR%;%GRADLE_LAUNCHER_JAR%

java -classpath "%CLASSPATH%" org.gradle.wrapper.GradleWrapperMain %*
endlocal

