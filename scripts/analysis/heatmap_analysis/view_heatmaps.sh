#!/bin/bash
# 在无GUI的Linux服务器上预览热力图的工具脚本

RESULTS_DIR="results/manipulation_analysis"

echo "🖼️ 操纵行为检测热力图预览工具"
echo "=================================="

if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ 结果目录不存在: $RESULTS_DIR"
    echo "请先运行: python scripts/analysis/manipulation_detection_heatmap.py"
    exit 1
fi

# 检查是否有图片文件
png_files=$(find "$RESULTS_DIR" -name "*.png" | wc -l)
if [ $png_files -eq 0 ]; then
    echo "❌ 未找到PNG文件"
    exit 1
fi

echo "📊 找到 $png_files 个热力图文件:"
ls -la "$RESULTS_DIR"/*.png

echo ""
echo "📝 查看方法："
echo "1. 下载到本地查看："
echo "   scp $(whoami)@$(hostname):\"$(pwd)/$RESULTS_DIR\"/*.png ./"
echo ""
echo "2. 如果安装了fim（终端图片查看器）："
echo "   fim -a \"$RESULTS_DIR/hourly_manipulation_heatmap.png\""
echo ""
echo "3. 如果安装了jp2a（ASCII艺术转换器）："
echo "   jp2a --color \"$RESULTS_DIR/hourly_manipulation_heatmap.png\""
echo ""
echo "4. 简单文件信息："

for file in "$RESULTS_DIR"/*.png; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file")
        echo "   📸 $filename: ${size} bytes"
    fi
done

echo ""
echo "✅ 所有热力图已成功生成并保存" 