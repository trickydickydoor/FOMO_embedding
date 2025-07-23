import re
from typing import List, Dict, Any, Tuple

class TextSplitter:
    """文本分割器，将长文本分割成适合向量化的小块"""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 separators: List[str] = None):
        """
        初始化文本分割器
        
        Args:
            chunk_size: 每个块的最大字符数
            chunk_overlap: 块之间的重叠字符数
            separators: 分割符列表，按优先级排序
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 默认分割符，按优先级排序
        if separators is None:
            self.separators = [
                "\n\n",  # 段落分割
                "\n",    # 行分割
                "。",     # 句子分割
                "！",     # 感叹句
                "？",     # 疑问句
                "；",     # 分号
                "，",     # 逗号
                " ",     # 空格
                ""       # 字符级分割
            ]
        else:
            self.separators = separators
    
    def split_text(self, text: str) -> List[Dict[str, Any]]:
        """
        分割文本为多个块
        
        Args:
            text: 要分割的文本
            
        Returns:
            文本块列表，每个块包含文本和位置信息
        """
        if not text:
            return []
        
        # 清理文本
        text = self._clean_text(text)
        
        # 按行分割以便计算行号
        lines = text.split('\n')
        
        # 重新组合文本但保留换行符位置信息
        char_to_line_map = self._build_char_line_map(lines)
        full_text = '\n'.join(lines)
        
        # 执行分割
        chunks = self._split_text_recursive(full_text, self.separators)
        
        # 为每个块添加位置信息
        result = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            # 找到这个块在原文中的位置
            start_pos = full_text.find(chunk)
            end_pos = start_pos + len(chunk) - 1
            
            # 转换为行号
            start_line = self._get_line_number(start_pos, char_to_line_map)
            end_line = self._get_line_number(end_pos, char_to_line_map)
            
            result.append({
                'text': chunk.strip(),
                'chunk_id': i,
                'char_start': start_pos,
                'char_end': end_pos,
                'line_start': start_line,
                'line_end': end_line,
                'length': len(chunk.strip())
            })
        
        return result
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 统一换行符
        text = re.sub(r'\r\n|\r', '\n', text)
        
        # 移除多余的空白行
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # 移除行首行尾空白
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def _build_char_line_map(self, lines: List[str]) -> Dict[int, int]:
        """构建字符位置到行号的映射"""
        char_to_line = {}
        char_pos = 0
        
        for line_num, line in enumerate(lines, 1):
            # 当前行的所有字符都对应这个行号
            for _ in range(len(line) + 1):  # +1 for newline
                char_to_line[char_pos] = line_num
                char_pos += 1
        
        return char_to_line
    
    def _get_line_number(self, char_pos: int, char_to_line_map: Dict[int, int]) -> int:
        """根据字符位置获取行号"""
        # 找到最接近的字符位置
        available_positions = sorted(char_to_line_map.keys())
        
        for pos in reversed(available_positions):
            if pos <= char_pos:
                return char_to_line_map[pos]
        
        return 1  # 默认第一行
    
    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """递归分割文本"""
        if len(text) <= self.chunk_size:
            return [text]
        
        # 尝试用当前分割符分割
        separator = separators[0] if separators else ""
        
        if separator == "":
            # 字符级分割
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i:i + self.chunk_size]
                chunks.append(chunk)
            return chunks
        
        # 用分割符分割
        splits = text.split(separator)
        
        if len(splits) == 1:
            # 当前分割符无效，尝试下一个
            if len(separators) > 1:
                return self._split_text_recursive(text, separators[1:])
            else:
                # 没有更多分割符，强制按大小分割
                return self._split_text_recursive(text, [""])
        
        # 重新组合分割结果
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # 测试添加这个分割后是否超过大小限制
            test_chunk = current_chunk + separator + split if current_chunk else split
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # 当前块已满，保存并开始新块
                if current_chunk:
                    chunks.append(current_chunk)
                
                # 如果单个分割太大，需要进一步分割
                if len(split) > self.chunk_size:
                    sub_chunks = self._split_text_recursive(split, separators[1:] if len(separators) > 1 else [""])
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)
        
        # 处理重叠
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)
        
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """为块添加重叠内容"""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # 从前一个块的末尾取重叠内容
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
            
            # 添加到当前块的开头
            overlapped_chunk = overlap_text + current_chunk
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks