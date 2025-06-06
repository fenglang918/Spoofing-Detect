#!/bin/bash
# 热力图分析统一启动脚本
# Unified Heatmap Analysis Launcher

# 默认参数
DATA_ROOT="/home/ma-user/code/fenglang/Spoofing Detect/data"
MODEL_PATH="results/trained_models/spoofing_model_Enhanced_undersample_Ensemble.pkl"
OUTPUT_DIR="results/heatmap_analysis"
BATCH_SIZE=30000
MAX_WORKERS=20
ANOMALY_WORKERS=20

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 显示帮助信息
show_help() {
    echo -e "${BLUE}热力图分析统一启动脚本${NC}"
    echo "=============================="
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help           显示此帮助信息"
    echo "  -d, --data DIR       数据根目录 (默认: $DATA_ROOT)"
    echo "  -m, --model FILE     模型文件路径 (默认: $MODEL_PATH)"
    echo "  -o, --output DIR     输出目录 (默认: $OUTPUT_DIR)"
    echo "  -b, --batch SIZE     批次大小 (默认: $BATCH_SIZE)"
    echo "  -w, --workers NUM    最大线程数 (默认: $MAX_WORKERS)"
    echo "  -a, --anomaly NUM    异常检测线程数 (默认: $ANOMALY_WORKERS)"
    echo "  --benchmark          运行性能基准测试"
    echo "  --quick              快速分析模式"
    echo "  --no-model           不使用模型，仅统计异常检测"
    echo ""
    echo "示例:"
    echo "  $0 --quick                          # 快速分析"
    echo "  $0 --benchmark                      # 性能测试"
    echo "  $0 -d /path/to/data --no-model      # 无模型分析"
    echo "  $0 -w 32 -a 24                      # 高性能配置"
}

# 检查依赖
check_dependencies() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    if [ ! -f "$script_dir/manipulation_detection_heatmap.py" ]; then
        echo -e "${RED}❌ 主分析脚本不存在${NC}"
        exit 1
    fi
    
    if [ ! -f "$script_dir/benchmark_tools.py" ]; then
        echo -e "${RED}❌ 基准测试脚本不存在${NC}"
        exit 1
    fi
}

# 检查系统资源
check_system() {
    local cpu_cores=$(nproc)
    local mem_gb=$(free -g | awk '/^Mem:/{print $2}')
    
    echo -e "${BLUE}💻 系统信息:${NC}"
    echo "  CPU核心数: $cpu_cores"
    echo "  内存: ${mem_gb}GB"
    
    # 自动调整线程数
    if [ $MAX_WORKERS -gt $cpu_cores ]; then
        MAX_WORKERS=$cpu_cores
        echo -e "${YELLOW}⚠️ 调整数据加载线程数为: $MAX_WORKERS${NC}"
    fi
    
    if [ $ANOMALY_WORKERS -gt $cpu_cores ]; then
        ANOMALY_WORKERS=$cpu_cores
        echo -e "${YELLOW}⚠️ 调整异常检测线程数为: $ANOMALY_WORKERS${NC}"
    fi
}

# 运行主分析
run_main_analysis() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local use_model="$1"
    
    echo -e "\n${BLUE}🚀 启动热力图分析${NC}"
    echo "========================="
    
    # 构建命令
    local cmd="python \"$script_dir/manipulation_detection_heatmap.py\""
    cmd="$cmd --data_root \"$DATA_ROOT\""
    cmd="$cmd --output_dir \"$OUTPUT_DIR\""
    cmd="$cmd --batch_size $BATCH_SIZE"
    cmd="$cmd --max_workers $MAX_WORKERS"
    cmd="$cmd --anomaly_workers $ANOMALY_WORKERS"
    
    if [ "$use_model" = "true" ] && [ -f "$MODEL_PATH" ]; then
        cmd="$cmd --model_path \"$MODEL_PATH\""
        echo -e "${GREEN}✅ 使用模型: $MODEL_PATH${NC}"
    else
        echo -e "${YELLOW}⚠️ 不使用模型，仅统计异常检测${NC}"
    fi
    
    echo -e "\n${BLUE}执行命令:${NC}"
    echo "$cmd"
    
    # 记录开始时间
    local start_time=$(date +%s)
    
    # 执行命令
    eval $cmd
    local exit_code=$?
    
    # 记录结束时间
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        echo -e "\n${GREEN}🎉 热力图分析完成！${NC}"
        echo -e "${BLUE}总耗时: $duration 秒${NC}"
        
        # 显示输出文件
        if [ -d "$OUTPUT_DIR" ]; then
            echo -e "\n${BLUE}📁 生成的文件:${NC}"
            ls -lh "$OUTPUT_DIR"
        fi
    else
        echo -e "\n${RED}❌ 分析失败，退出代码: $exit_code${NC}"
        exit $exit_code
    fi
}

# 运行基准测试
run_benchmark() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    echo -e "\n${BLUE}🧪 启动性能基准测试${NC}"
    echo "========================="
    
    local cmd="python \"$script_dir/benchmark_tools.py\""
    cmd="$cmd --samples 20000"
    cmd="$cmd --stocks 30"
    cmd="$cmd --output_dir \"$OUTPUT_DIR/benchmark\""
    
    echo -e "${BLUE}执行命令:${NC}"
    echo "$cmd"
    
    eval $cmd
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "\n${GREEN}🎉 基准测试完成！${NC}"
    else
        echo -e "\n${RED}❌ 基准测试失败${NC}"
        exit $exit_code
    fi
}

# 主程序
main() {
    local mode="full"
    local use_model="true"
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -d|--data)
                DATA_ROOT="$2"
                shift 2
                ;;
            -m|--model)
                MODEL_PATH="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -b|--batch)
                BATCH_SIZE="$2"
                shift 2
                ;;
            -w|--workers)
                MAX_WORKERS="$2"
                shift 2
                ;;
            -a|--anomaly)
                ANOMALY_WORKERS="$2"
                shift 2
                ;;
            --benchmark)
                mode="benchmark"
                shift
                ;;
            --quick)
                mode="quick"
                shift
                ;;
            --no-model)
                use_model="false"
                shift
                ;;
            *)
                echo -e "${RED}未知参数: $1${NC}"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 检查依赖和系统
    check_dependencies
    check_system
    
    # 检查数据目录
    if [ ! -d "$DATA_ROOT" ]; then
        echo -e "${RED}❌ 数据目录不存在: $DATA_ROOT${NC}"
        exit 1
    fi
    
    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"
    
    echo -e "${BLUE}📋 运行参数:${NC}"
    echo "  数据目录: $DATA_ROOT"
    echo "  输出目录: $OUTPUT_DIR"
    echo "  批次大小: $BATCH_SIZE"
    echo "  最大线程数: $MAX_WORKERS"
    echo "  异常检测线程数: $ANOMALY_WORKERS"
    echo "  运行模式: $mode"
    echo "  使用模型: $use_model"
    
    # 执行相应模式
    case $mode in
        "benchmark")
            run_benchmark
            ;;
        "quick"|"full")
            run_main_analysis "$use_model"
            ;;
        *)
            echo -e "${RED}未知运行模式: $mode${NC}"
            exit 1
            ;;
    esac
    
    echo -e "\n${GREEN}✅ 所有任务完成！${NC}"
}

# 执行主程序
main "$@" 