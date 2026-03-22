#!/usr/bin/env sh
APP_HOME=$(cd "$(dirname "$0")" && pwd)
WRAPPER_JAR="$APP_HOME/gradle/wrapper/gradle-wrapper.jar"
WRAPPER_SHARED_JAR="$APP_HOME/gradle/wrapper/gradle-wrapper-shared.jar"
CLASSPATH="$WRAPPER_JAR:$WRAPPER_SHARED_JAR"
exec java -classpath "$CLASSPATH" org.gradle.wrapper.GradleWrapperMain "$@"

