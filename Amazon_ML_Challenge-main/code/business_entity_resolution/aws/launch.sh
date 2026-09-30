#!/bin/bash
# Launch a self-terminating spot instance that runs the pipeline.
# usage: aws/launch.sh <instance-type> "<stages>" <run-name> <max-hours>
set -euo pipefail
TYPE=${1:-r7i.16xlarge}; STAGES=${2:-"prep stage_a stage1 stage_b train2 predict"}; RUN=${3:-v1}; HOURS=${4:-6}
BUCKET=${ER_BUCKET:?set ER_BUCKET}
# ER_MARKET=spot (default) or ondemand
if [ "${ER_MARKET:-spot}" = "spot" ]; then
  MARKET="--instance-market-options MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}"
else MARKET=""; fi; PROFILE=${ER_PROFILE:-er-challenge-ec2}
AMI=$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 --query Parameter.Value --output text)
UD=$(sed -e "s|__BUCKET__|$BUCKET|; s|__STAGES__|$STAGES|; s|__RUN__|$RUN|; s|__HOURS__|$HOURS|" "$(dirname "$0")/bootstrap.sh")
aws ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --count 1 \
  --iam-instance-profile Name="$PROFILE" \
  $MARKET \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=er-$RUN},{Key=project,Value=az-ml-challenge}]" \
  --user-data "$UD" --query "Instances[0].InstanceId" --output text
