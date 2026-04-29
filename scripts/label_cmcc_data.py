"""
为中国移动知识库数据添加分类标签
每个条目根据内容自动归类，生成分类→内容的映射
"""
import os
import re
import json

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cmcc_300_knowledge.txt")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge_data", "cmcc_knowledge_labeled.json")

# 分类规则：(类别名, 匹配关键词列表)
CATEGORIES = [
    ("5G与6G", ["5G", "5G-A", "6G", "RedCap", "超级上行", "通感一体", "载波聚合", "毫米波",
                "Massive MIMO", "波束赋形", "上下行解耦", "无源物联", "TSN", "WDM"]),
    ("算力网络", ["算力", "算网", "边缘算力", "GPU", "CPU", "异构算力", "算力调度", "算网大脑",
                 "东数西算", "算力交易", "算力节点"]),
    ("移动云", ["移动云", "云桌面", "云专线", "云切片", "云网融合", "容器云", "微服务", "多云",
               "中心云", "区域云", "边缘云", "专属云", "私有云", "混合云", "上云"]),
    ("物联网", ["物联网", "IoT", "NB-IoT", "Cat.1", "LoRa", "OneNET", "物联卡", "物联终端",
               "设备管理", "M2M", "RFID", "时序数据"]),
    ("AI与大模型", ["AI", "大模型", "九天", "人工智能", "视觉检测", "语义", "节能算法",
                  "AI中台", "AI节能", "AI视频", "画质增强", "智能运维"]),
    ("宽带与光通信", ["宽带", "FTTR", "PON", "光宽带", "光纤", "Wi-Fi", "WiFi", "全屋",
                    "XGS-PON", "OTN", "波分复用", "光网络"]),
    ("专网与切片", ["专网", "切片", "SPN", "硬隔离", "软隔离", "专属带宽", "工业5G",
                   "网络切片", "专线", "防爆基站", "化工园区"]),
    ("网络安全", ["安全", "加密", "反诈", "等保", "隐私", "防火墙", "审计", "认证",
                "数据安全", "容灾", "身份认证", "权限", "防泄露"]),
    ("大数据", ["大数据", "数据中台", "数据治理", "数据脱敏", "数据共享", "数据分析",
               "热力图", "联邦学习", "隐私计算"]),
    ("政企服务", ["政企", "政务", "党政", "国企", "数字化方案", "上云迁移", "专属运维",
                 "行业定制", "解决方案", "政务云", "档案"]),
    ("国际通信", ["国际", "跨境", "海外", "漫游", "海缆", "全球", "出海", "境外"]),
    ("智慧行业", ["智慧", "数字乡村", "智慧医疗", "智慧交通", "智慧电力", "智慧校园",
                 "智慧养老", "矿山", "金融", "V2X", "车联网", "远程"]),
    ("应急与保障", ["应急", "灾", "防汛", "抗震", "保障", "备用", "储能", "冗余",
                   "双路由", "备份", "不间断"]),
    ("核心网与传输", ["核心网", "承载网", "骨干", "路由", "IPv6", "NFV", "SDN", "网络虚拟化",
                     "传输网", "CDN", "边缘节点"]),
    ("绿色低碳", ["节能", "低碳", "绿色", "PUE", "光伏", "液冷", "功耗", "能耗", "碳排放"]),
    ("卫星与天地一体", ["卫星", "北斗", "星地", "低轨", "天通", "天地一体"]),
]

def categorize(text):
    """根据文本内容自动分类"""
    for category, keywords in CATEGORIES:
        for kw in keywords:
            if kw in text:
                return category
    return "综合业务"

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    labeled = []
    for i, line in enumerate(lines):
        category = categorize(line)
        labeled.append({
            "id": i + 1,
            "category": category,
            "content": line
        })

    # 统计分类分布
    dist = {}
    for item in labeled:
        dist[item["category"]] = dist.get(item["category"], 0) + 1

    print("=== 分类分布 ===")
    for cat, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} 条")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(labeled, f, ensure_ascii=False, indent=2)

    print(f"\n已生成 {len(labeled)} 条带标签数据 → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
