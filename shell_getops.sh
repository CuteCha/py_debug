#!/bin/bash

func() {
  echo "Usage:"
  echo "bash ntools.sh -j [init|build|help|kill|venus] -c [start|kill] -y [conf.yaml] -t [date_id]"
  exit 0
}

while getopts 'j:c:y:t' OPT; do
  case $OPT in
  j) JOB="$OPTARG" ;;
  c) COMMAND="$OPTARG" ;;
  y) YMAL_PATH="$OPTARG" ;;
  t) DATE_ID="$OPTARG" ;;
  ?) func ;;
  esac
done

echo "JOB=${JOB}"
echo "COMMAND=${COMMAND}"
echo "YMAL_PATH=${YMAL_PATH}"
echo "DATE_ID=${DATE_ID}"

if [[ -n "$YMAL_PATH" ]]; then
  echo "YMAL_PATH is not empty"
fi

if [[ -f "$YMAL_PATH" ]]; then
  echo "YMAL_PATH exist"
fi

if [[ ! -n "$YMAL_PATH" ]]; then
  echo "YMAL_PATH is empty"
fi

if [[ ! -f "$YMAL_PATH" ]]; then
  echo "YMAL_PATH not exist"
fi

if [[ -z "$DATE_ID" ]]; then
  echo "DATE_ID is empty"
  DATE_ID=$(date +%Y%m%d)
fi

echo "------"
echo "JOB=${JOB}"
echo "COMMAND=${COMMAND}"
echo "YMAL_PATH=${YMAL_PATH}"
echo "DATE_ID=${DATE_ID}"

<<'COMMENT'
echo "-------"
for arg in "$*"
do
    echo $arg
done

echo "---------------"

for arg in "$@"
do
    echo $arg
done
COMMENT
