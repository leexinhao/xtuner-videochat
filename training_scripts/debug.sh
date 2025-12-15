set -ex

master_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
all_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo "All nodes used: ${all_nodes}"
echo "Master node ${master_node}"

# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$master_node" hostname --ip-address | awk '{print $1}')
# head_node_ip=$(ssh -o StrictHostKeyChecking=no "$master_node" hostname --ip-address | awk '{print $1}')
# head_node_ip=$master_node
# 步骤1：将节点名按 "-" 分割为数组
IFS='-' read -ra parts <<< "$master_node"
# 步骤2：取数组最后4个元素（索引：-4、-3、-2、-1）
last_four=("${parts[@]: -4}")
# 步骤3：用 "." 连接这4个元素
head_node_ip="${last_four[0]}.${last_four[1]}.${last_four[2]}.${last_four[3]}"


rdzv_endpoint="${head_node_ip}:${MASTER_PORT:-40000}"

echo "${rdzv_endpoint}"