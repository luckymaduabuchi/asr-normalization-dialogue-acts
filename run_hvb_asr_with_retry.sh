#!/usr/bin/env bash
cd /home/vm-user/Desktop/connl
for i in $(seq 1 30); do
  echo "=== attempt $i ==="
  python3 -u hvb_generate_real_asr.py
  if [ $? -eq 0 ]; then
    echo "=== ASR generation finished successfully ==="
    exit 0
  fi
  echo "=== crashed, retrying in 20s (resumable via checkpoints) ==="
  sleep 20
done
echo "=== gave up after 30 attempts ==="
exit 1
