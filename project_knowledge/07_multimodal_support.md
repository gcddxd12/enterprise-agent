# 多模态支持

## 概述
多模态支持模块（`multimodal_support.py`）提供图像处理、文档解析和多模态知识库功能。**注意：在v2.0标准ReAct架构中，多模态工具（image_analysis、document_processing、file_upload_processing）已从Agent工具列表中移除**，该模块作为独立的工具库保留，可在需要时按需注册。

## 1. 模块定位（v2.0）

**文件**: [multimodal_support.py](e:\my_multi_agent\multimodal_support.py)

**当前状态**:
- 模块代码保留完整，可独立使用
- **未注册到 `AGENT_TOOLS`** 列表中（主Agent只包含4个核心工具）
- 如需启用多模态功能，在 `AGENT_TOOLS` 中添加对应的 `@tool` 函数即可
- 演示文件 `demo_multimodal_features.py` 和测试文件 `test_multimodal_integration.py` 已删除

**为什么从Agent中移除**:
- 当前客服场景以文本查询为主
- 模拟实现（mock OCR等）在生产中价值有限
- 减少LLM可选择工具的数量，提升决策准确性
- 遵循"用不到的代码不留"的清理原则

**如何重新启用**:
```python
# 在 langgraph_agent_with_memory.py 中
from multimodal_support import MultimodalTools

multimodal_tools = MultimodalTools()

@tool
def image_analysis(image_path: str) -> str:
    """分析图像内容"""
    return multimodal_tools.analyze_image(image_path)

# 加入工具列表
AGENT_TOOLS = [..., image_analysis]
```

## 2. 媒体类型系统

### 2.1 媒体类型定义

**文件**: [multimodal_support.py](e:\my_multi_agent\multimodal_support.py)

```python
class MediaType(Enum):
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"
    DOCX = "docx"
    EXCEL = "excel"
    AUDIO = "audio"
    VIDEO = "video"
    UNKNOWN = "unknown"

@dataclass
class MediaContent:
    content_type: MediaType
    file_path: Optional[str] = None
    raw_data: Optional[bytes] = None
    text_content: Optional[str] = None
    metadata: Dict[str, Any] = None
```

### 2.2 媒体类型检测器

```python
class MediaDetector:
    @staticmethod
    def detect_media_type(file_path: str) -> MediaType:
        """基于MIME类型 + 扩展名检测文件类型"""
```

## 3. 图像处理

### 3.1 ImageProcessor

```python
class ImageProcessor:
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock

    def extract_text_from_image(self, image_path: str) -> str:
        """从图像中提取文字（当前为模拟OCR）"""

    def describe_image(self, image_path: str) -> str:
        """描述图像内容"""

    def extract_metadata(self, image_path: str) -> Dict:
        """提取图像元数据（尺寸、格式等）"""
```

**模拟vs实际**: 当前使用mock实现（基于文件名的预定义内容匹配），实际部署时可替换为Tesseract OCR、CLIP等真实模型。

## 4. 文档处理

### 4.1 DocumentProcessor

```python
class DocumentProcessor:
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock

    def extract_text_from_pdf(self, pdf_path: str) -> str:
    def extract_text_from_docx(self, docx_path: str) -> str:
    def extract_text_from_excel(self, excel_path: str) -> str:
```

**实际部署技术栈**:
- PDF: `pdfplumber` 或 `PyPDF2`
- Word: `python-docx`
- Excel: `openpyxl` 或 `pandas`

## 5. 多模态知识库

### 5.1 MultimodalKnowledgeBase

```python
class MultimodalKnowledgeBase:
    def __init__(self, storage_dir: str = "./multimodal_kb"):
        self.storage_dir = storage_dir
        self.media_index = {}

    def add_media_file(self, file_path: str) -> Optional[MediaContent]:
        """添加媒体文件到知识库"""

    def search_media(self, query: str, media_type=None) -> List[MediaContent]:
        """搜索媒体内容（文本匹配 + 文件名匹配）"""
```

## 6. 工具集合（MultimodalTools）

```python
class MultimodalTools:
    def __init__(self):
        self.knowledge_base = MultimodalKnowledgeBase()
        self.image_processor = ImageProcessor(use_mock=True)

    def process_uploaded_file(self, file_path: str) -> str:
    def search_multimodal_content(self, query: str) -> str:
    def analyze_image(self, image_path: str) -> str:
```

## 7. 模块架构

```
multimodal_support.py
├── MediaType / MediaContent     # 基础类型定义
├── MediaDetector               # 媒体类型检测
├── ImageProcessor              # 图像处理（OCR/描述/元数据）
├── DocumentProcessor           # 文档处理（PDF/Word/Excel）
├── MultimodalKnowledgeBase     # 多模态知识库存储+检索
└── MultimodalTools             # 统一工具集合（对外接口）
```

## 8. 学习总结

### 关键要点
1. 多模态模块作为独立工具库保留，不参与主Agent工作流
2. 当前使用模拟实现，便于快速原型开发
3. 如需启用，只需定义 `@tool` + 注册到 `AGENT_TOOLS`
4. 实际部署时替换模拟实现为真实模型即可

### 最佳实践
1. **渐进式实现**: mock → 真实模型替换，降低开发风险
2. **按需注册**: 不使用的工具不要注册到Agent，减少LLM决策负担
3. **模块独立**: 多模态模块与主Agent解耦，互不影响

---

**相关文件**:
- [multimodal_support.py](e:\my_multi_agent\multimodal_support.py) — 多模态支持核心实现
- [langgraph_agent_with_memory.py](e:\my_multi_agent\langgraph_agent_with_memory.py) — 主Agent（多模态工具未在此注册）

**下一步学习**: 工程化实践 →
