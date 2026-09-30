#!/bin/bash
# EC2 user-data: run the ER pipeline end-to-end, push results to S3, then self-terminate.
# Placeholders __BUCKET__, __STAGES__, __RUN__, __HOURS__ are filled by launch.sh.
set -euxo pipefail
exec > >(tee /var/log/er-run.log) 2>&1
BUCKET=__BUCKET__; STAGES="__STAGES__"; RUN=__RUN__
# hard cost cap: power off (=> terminate) after N hours no matter what
shutdown -h +$((__HOURS__*60)) || true
dnf install -y python3.12 python3.12-pip git
cd /opt && git clone https://github.com/Zayaan3019/AZ_ML_Challenge.git repo && cd repo
python3.12 -m venv .venv && .venv/bin/pip install -q -r code/business_entity_resolution/requirements.txt
mkdir -p student_resource/dataset work
aws s3 sync s3://$BUCKET/dataset student_resource/dataset --only-show-errors
aws s3 sync s3://$BUCKET/work work --only-show-errors || true   # resume from checkpoints
export ER_WORKERS=$(( $(nproc) - 2 ))
cd code/business_entity_resolution/src
sync_up() { aws s3 sync /opt/repo/work s3://$BUCKET/work --only-show-errors --exclude "*_s[123].parquet" || true;
            aws s3 sync /opt/repo/output s3://$BUCKET/output --only-show-errors || true;
            aws s3 cp /var/log/er-run.log s3://$BUCKET/logs/$RUN-$(date +%s).log || true; }
trap 'sync_up; shutdown -h now' EXIT
for st in $STAGES; do ../../../.venv/bin/python -m er.run $st --run $RUN; sync_up; done
