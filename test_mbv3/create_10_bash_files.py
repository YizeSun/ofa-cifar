base_script = """#!/bin/bash
#PBS -N evaluate_cifar10_part_{part_id}
#PBS -l select=2:node_type=mi300a:mpiprocs=4:ncpus=96
#PBS -l walltime=24:00:00
#PBS -o /lustre/hpe/ws12/ws12.a/ws/xmuyzsun-WK0/outputs/eval_part_{part_id}.log
#PBS -e /lustre/hpe/ws12/ws12.a/ws/xmuyzsun-WK0/outputs/eval_part_{part_id}.err

cd $WK0/ofa-cifar/test_mbv3/
echo $WK0
echo $(date) >> /lustre/hpe/ws12/ws12.a/ws/xmuyzsun-WK0/outputs/eval_part_{part_id}.log
echo $(date) >> /lustre/hpe/ws12/ws12.a/ws/xmuyzsun-WK0/outputs/eval_part_{part_id}.err

module use /opt/hlrs/testing/ai-frameworks/modulefiles
module load pytorch
source ~/.venv/ofa-cifar-hunter/bin/activate

export DEEPSPEED_HOSTFILE=deepspeed.hostfile
WORLD_SIZE=`cat $PBS_NODEFILE | wc -l`

sort $PBS_NODEFILE | awk -F'.' '{{print $1}}' | uniq | while read node; do
    echo "$node slots=4"
done > "$DEEPSPEED_HOSTFILE"
export HEAD_NODE_IP=$(head -n 1 $PBS_NODEFILE | awk -F'.' '{{print $1}}')
export MASTER_PORT=29500
export MASTER_ADDR=$HEAD_NODE_IP

if [ -f $PBS_NODEFILE ]; then
  > $DEEPSPEED_HOSTFILE
  sort $PBS_NODEFILE | uniq | while read host; do
    short_name=$(echo $host | cut -d '.' -f 1)
    echo "$short_name slots=4" >> $DEEPSPEED_HOSTFILE
  done
else
  echo "Error: PBS_NODEFILE not found!"
  exit 1
fi

GRAPH_DIR="${{WK0}}ofa-cifar/test_mbv3/graphs"
SCRIPT="evaluate_mbv3_cifar10.py"
python test_final_graphs_cifar10.py

# Processing files: {file_list}
{loop_block}

# Final Summary
echo -e "\\n📊 Final Summary:"
python $SCRIPT --input {input_summary}
"""

for part_id in range(10):
    start = part_id * 3
    end = start + 3
    files = [f"$GRAPH_DIR/cifar10_graph_{i:03d}.json" for i in range(start, end)]
    file_list = " ".join(files)
    input_summary = " ".join(files)

    loop = "\n".join([
        f'for graph_file in {file_list}; do',
        '    echo "📂 Evaluating $graph_file"',
        '    mpirun -np $WORLD_SIZE \\',
        '            --ppn 4 \\',
        '            --genvall \\',
        '            --cpu-bind list:0-23:24-47:48-71:72-95 \\',
        '            --gpu-bind none \\',
        '            --no-transfer \\',
        '            -genv WORLD_SIZE=$WORLD_SIZE \\',
        '            -genv LOCAL_SIZE=4 \\',
        '            python -u /opt/hlrs/testing/ai-frameworks/DeepSpeed/deepspeed/launcher/launcher_helper.py \\',
        '            --launcher mpich \\',
        '            $SCRIPT --input "$graph_file"',
        'done'
    ])

    bash_content = base_script.format(
        part_id=part_id,
        file_list=file_list,
        loop_block=loop,
        input_summary=input_summary
    )

    with open(f"evaluate_cifar10_part_{part_id}.pbs", "w") as f:
        f.write(bash_content)
