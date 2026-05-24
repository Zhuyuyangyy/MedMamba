#!/bin/bash
# =============================================================================
# MedMamba-Guard Start Script - 交互式启动脚本
# 支持: info / benchmark / train / serve / demo / guard / risk-analysis / ablation
# =============================================================================

# 确保使用bash运行
if [ -z "$BASH_VERSION" ]; then
    exec bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${CYAN}==============================================${NC}"
echo -e "${CYAN}  MedMamba-Guard 可信推理系统${NC}"
echo -e "${CYAN}  基于SSM状态轨迹的医学影像可信框架${NC}"
echo -e "${CYAN}==============================================${NC}"
echo ""

# 检测Python环境
if command -v python3 > /dev/null 2>&1; then
    PYTHON="python3"
elif command -v python > /dev/null 2>&1; then
    PYTHON="python"
else
    echo -e "${RED}Error: Python not found!${NC}"
    exit 1
fi

echo -e "${GREEN}Using: $PYTHON${NC}"

# 检查依赖
echo ""
echo "检查依赖..."

check_pkg() {
    if $PYTHON -c "import $1" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "  ${RED}✗${NC} $1 (missing)"
        return 1
    fi
}

MISSING_DEPS=()
check_pkg "torch" || MISSING_DEPS+=(torch)
check_pkg "einops" || MISSING_DEPS+=(einops)
check_pkg "numpy" || MISSING_DEPS+=(numpy)
check_pkg "Pillow" || MISSING_DEPS+=(Pillow)
check_pkg "fastapi" || MISSING_DEPS+=(fastapi)
check_pkg "uvicorn" || MISSING_DEPS+=(uvicorn)

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    echo ""
    echo -e "${YELLOW}安装缺失依赖: ${MISSING_DEPS[*]}${NC}"
    $PYTHON -m pip install -q "${MISSING_DEPS[@]}" 2>/dev/null || \
    pip install -q "${MISSING_DEPS[@]}"
fi

echo ""
echo "=============================================="
echo "  可用命令:"
echo "=============================================="
echo ""
echo "  info           - 查看模型架构信息"
echo "  benchmark      - 测试模型性能"
echo "  train          - 训练模型"
echo "  train-guard    - 训练MedMamba-Guard模型"
echo "  serve          - 启动API服务 (端口8866)"
echo "  demo           - 启动Web演示界面 (前端+后端)"
echo "  guard-demo     - 风险评估演示（无需训练）"
echo "  risk-analysis  - 风险-错误相关性分析"
echo "  ablation       - 消融实验"
echo ""
echo "  或直接运行: ./start.sh [command]"
echo "=============================================="

# 解析命令行参数
MODE="${1:-menu}"

show_menu() {
    echo ""
    echo "请选择操作:"
    echo "  1) info           - 查看模型架构信息"
    echo "  2) benchmark      - 测试模型性能"
    echo "  3) train          - 训练模型"
    echo "  4) train-guard    - 训练MedMamba-Guard模型"
    echo "  5) serve          - 启动API服务"
    echo "  6) demo           - 启动Web演示界面"
    echo "  7) guard-demo     - 风险评估演示"
    echo "  8) risk-analysis  - 风险-错误相关性分析"
    echo "  9) ablation       - 消融实验"
    echo "  q) 退出"
    echo ""
    read -p "请输入选项 [1-9, q]: " choice
    echo ""
    case "$choice" in
        1) MODE="info" ;;
        2) MODE="benchmark" ;;
        3) MODE="train" ;;
        4) MODE="train-guard" ;;
        5) MODE="serve" ;;
        6) MODE="demo" ;;
        7) MODE="guard-demo" ;;
        8) MODE="risk-analysis" ;;
        9) MODE="ablation" ;;
        q|Q) echo "Goodbye!"; exit 0 ;;
        *) echo "无效选项"; show_menu; return ;;
    esac
}

if [ "$MODE" = "menu" ]; then
    show_menu
fi

case "$MODE" in
    info)
        echo -e "${CYAN}[INFO] 查看模型架构...${NC}"
        $PYTHON main.py --mode info
        ;;

    benchmark)
        echo -e "${CYAN}[BENCHMARK] 测试模型性能...${NC}"
        $PYTHON main.py --mode benchmark
        ;;

    train)
        echo -e "${CYAN}[TRAIN] 训练模型...${NC}"
        shift
        $PYTHON train_medmamba.py "$@"
        ;;

    train-guard)
        echo -e "${CYAN}[TRAIN-GUARD] 训练MedMamba-Guard模型...${NC}"
        shift
        $PYTHON train_medmamba_guard.py "$@"
        ;;

    serve)
        echo -e "${CYAN}[SERVE] 启动API服务...${NC}"
        echo ""
        echo -e "${GREEN}API服务地址: http://localhost:8866${NC}"
        echo -e "${GREEN}API文档:     http://localhost:8866/docs${NC}"
        echo -e "${GREEN}Guard接口:   http://localhost:8866/predict${NC}"
        echo ""
        $PYTHON src/api/server.py
        ;;

    demo)
        echo -e "${CYAN}[DEMO] 启动Web演示界面...${NC}"
        echo ""

        # 检查端口是否占用
        if lsof -Pi :8866 -sTCP:LISTEN -t &>/dev/null; then
            echo -e "${YELLOW}端口8866已被占用，尝试关闭...${NC}"
            lsof -Pi :8866 -sTCP:LISTEN -t | xargs kill -9 2>/dev/null || true
            sleep 1
        fi

        # 启动后端服务
        echo -e "${GREEN}启动API服务 (后台)...${NC}"
        $PYTHON src/api/server.py &
        API_PID=$!

        # 等待服务启动
        sleep 3

        # 检查服务是否启动成功
        if ! kill -0 $API_PID 2>/dev/null; then
            echo -e "${RED}API服务启动失败${NC}"
            exit 1
        fi

        echo -e "${GREEN}API服务已启动 (PID: $API_PID)${NC}"
        echo ""
        echo -e "${CYAN}==============================================${NC}"
        echo -e "${CYAN}  Web演示界面已启动${NC}"
        echo -e "${CYAN}==============================================${NC}"
        echo ""
        echo -e "${GREEN}  前端地址: file://${SCRIPT_DIR}/frontend/index.html${NC}"
        echo -e "${GREEN}  API地址:  http://localhost:8866${NC}"
        echo -e "${GREEN}  API文档:  http://localhost:8866/docs${NC}"
        echo ""
        echo -e "${YELLOW}  按 Ctrl+C 停止服务${NC}"
        echo ""

        # 检测系统并打开浏览器
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sleep 1
            open "file://${SCRIPT_DIR}/frontend/index.html" 2>/dev/null || true
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux
            xdg-open "file://${SCRIPT_DIR}/frontend/index.html" 2>/dev/null || true
        elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
            # Windows Git Bash
            start "file://${SCRIPT_DIR}/frontend/index.html" 2>/dev/null || true
        fi

        # 等待信号
        trap "echo '正在停止服务...'; kill $API_PID 2>/dev/null; exit 0" INT TERM

        wait $API_PID
        ;;

    guard-demo)
        echo -e "${CYAN}[GUARD-DEMO] 风险评估演示...${NC}"
        echo ""
        echo -e "${GREEN}启动API服务进行风险评估演示...${NC}"
        
        # 检查端口
        if lsof -Pi :8866 -sTCP:LISTEN -t &>/dev/null; then
            echo -e "${YELLOW}端口8866已被占用...${NC}"
            lsof -Pi :8866 -sTCP:LISTEN -t | xargs kill -9 2>/dev/null || true
            sleep 1
        fi
        
        # 启动服务
        $PYTHON src/api/server.py &
        API_PID=$!
        sleep 3
        
        if ! kill -0 $API_PID 2>/dev/null; then
            echo -e "${RED}API服务启动失败${NC}"
            exit 1
        fi
        
        echo -e "${GREEN}API服务已启动 (PID: $API_PID)${NC}"
        echo ""
        echo -e "${CYAN}MedMamba-Guard 风险评估演示${NC}"
        echo -e "${CYAN}================================${NC}"
        echo ""
        echo -e "${YELLOW}测试接口:${NC}"
        echo "  curl -X POST http://localhost:8866/predict -F 'file=@test.jpg'"
        echo "  curl -X POST http://localhost:8866/predict-risk-only -F 'file=@test.jpg'"
        echo ""
        echo -e "${GREEN}API文档: http://localhost:8866/docs${NC}"
        echo ""
        echo -e "${YELLOW}按 Ctrl+C 停止${NC}"
        
        wait $API_PID
        ;;

    risk-analysis)
        echo -e "${CYAN}[RISK-ANALYSIS] 风险-错误相关性分析...${NC}"
        echo ""
        echo -e "${YELLOW}运行风险-错误AUROC分析...${NC}"
        $PYTHON -c "
import sys
sys.path.insert(0, '.')
from src.evaluator import RiskErrorAnalyzer

analyzer = RiskErrorAnalyzer()
results = analyzer.run_auroc_analysis()
print('Risk-Error AUROC Results:')
for k, v in results.items():
    print(f'  {k}: {v:.4f}')
"
        ;;

    ablation)
        echo -e "${CYAN}[ABLATION] 消融实验...${NC}"
        echo ""
        echo -e "${YELLOW}运行消融实验 (CTM/Cross-Scan/互证门控)...${NC}"
        $PYTHON -c "
import sys
sys.path.insert(0, '.')
from src.evaluator import AblationStudy

study = AblationStudy()
results = study.run()
print('Ablation Study Results:')
for config, metrics in results.items():
    print(f'  {config}:')
    for k, v in metrics.items():
        print(f'    {k}: {v:.4f}')
"
        ;;

    *)
        echo -e "${RED}未知命令: $MODE${NC}"
        echo "可用命令: info, benchmark, train, train-guard, serve, demo, guard-demo, risk-analysis, ablation"
        exit 1
        ;;
esac

echo ""
echo "Done!"