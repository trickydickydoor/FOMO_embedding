#!/usr/bin/env python3
"""
新闻Embedding处理脚本
定时从Supabase获取未处理的新闻，生成embedding并存储到Pinecone
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.supabase_handler import SupabaseHandler
from vector.pinecone_handler import PineconeHandler
from vector.text_embedder import TextEmbedder

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsEmbeddingProcessor:
    """处理新闻embedding的主类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化处理器
        
        Args:
            config_path: 配置文件路径，默认使用vector/embedding_config.json
        """
        # 设置配置路径
        if config_path:
            self.config_path = config_path
        else:
            self.config_path = project_root / "vector" / "embedding_config.json"
        
        # 初始化各个组件
        self.supabase = None
        self.pinecone = None
        self.embedder = None
        
        # 批处理设置
        self.batch_size = 50
        self.max_items_per_run = 1000  # 每次运行最多处理的条目数
        
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化所有组件"""
        logger.info("初始化组件...")
        
        # 初始化Supabase
        self.supabase = SupabaseHandler(
            config_file=str(self.config_path),
            log_callback=logger.info
        )
        
        # 初始化Pinecone
        self.pinecone = PineconeHandler(
            config_file=str(self.config_path),
            log_callback=logger.info
        )
        
        # 初始化Text Embedder
        self.embedder = TextEmbedder(
            config_file=str(self.config_path),
            log_callback=logger.info
        )
        
        # 检查连接状态
        if not self.supabase.client:
            raise Exception("无法连接到Supabase")
        if not self.pinecone.check_connection():
            raise Exception("无法连接到Pinecone")
        if not self.embedder.check_connection():
            raise Exception("无法连接到Gemini API")
        
        logger.info("所有组件初始化成功")
    
    def get_pending_news_items(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        获取待处理的新闻项
        
        Args:
            limit: 限制数量
            
        Returns:
            新闻项列表
        """
        try:
            # 查询embedding_status为'pending'的记录
            query = self.supabase.client.table('news_items').select('*').eq('embedding_status', 'pending')
            
            # 按created_at排序，优先处理旧的
            query = query.order('created_at', desc=False)
            
            # 限制数量
            if limit:
                query = query.limit(limit)
            else:
                query = query.limit(self.max_items_per_run)
            
            response = query.execute()
            
            logger.info(f"找到 {len(response.data)} 个待处理的新闻项")
            return response.data
            
        except Exception as e:
            logger.error(f"获取待处理新闻失败: {e}")
            return []
    
    def update_news_status(self, news_ids: List[int], status: str, vector_ids: Optional[List[str]] = None):
        """
        更新新闻的embedding状态
        
        Args:
            news_ids: 新闻ID列表
            status: 新状态
            vector_ids: Pinecone向量ID列表（可选）
        """
        try:
            # 准备更新数据
            update_data = {
                'embedding_status': status,
                'embedding_model': 'text-embedding-004'
            }
            
            # 如果是完成状态，添加时间戳
            if status == 'completed':
                update_data['embedded_at'] = datetime.utcnow().isoformat()
            
            # 批量更新
            for i, news_id in enumerate(news_ids):
                if vector_ids and i < len(vector_ids):
                    update_data['embedding_vector_id'] = vector_ids[i]
                
                try:
                    self.supabase.client.table('news_items').update(update_data).eq('id', news_id).execute()
                except Exception as e:
                    logger.error(f"更新新闻 {news_id} 状态失败: {e}")
            
            logger.info(f"成功更新 {len(news_ids)} 个新闻项的状态为 {status}")
            
        except Exception as e:
            logger.error(f"批量更新状态失败: {e}")
    
    def process_batch(self, news_items: List[Dict[str, Any]]) -> int:
        """
        处理一批新闻项
        
        Args:
            news_items: 新闻项列表
            
        Returns:
            成功处理的数量
        """
        if not news_items:
            return 0
        
        # 提取ID列表
        news_ids = [item['id'] for item in news_items]
        
        # 更新状态为processing
        self.update_news_status(news_ids, 'processing')
        
        try:
            # 过滤出有内容的新闻项
            items_with_content = [item for item in news_items if item.get('content')]
            if len(items_with_content) < len(news_items):
                logger.warning(f"过滤掉 {len(news_items) - len(items_with_content)} 个没有内容的新闻项")
            
            # 生成embeddings（现在支持文本分割）
            logger.info(f"开始生成 {len(items_with_content)} 个新闻项的embeddings...")
            embeddings, successful_chunks = self.embedder.generate_embeddings(items_with_content)
            
            if not embeddings:
                logger.warning("没有成功生成任何embeddings")
                # 恢复状态为pending
                self.update_news_status(news_ids, 'pending')
                return 0
            
            # 统计成功处理的新闻ID
            processed_news_ids = set()
            for chunk in successful_chunks:
                news_id = chunk['news_item']['id']
                processed_news_ids.add(news_id)
            
            logger.info(f"成功生成 {len(embeddings)} 个文本块向量，涉及 {len(processed_news_ids)} 篇新闻")
            
            # 上传到Pinecone
            logger.info(f"上传 {len(embeddings)} 个向量到Pinecone...")
            vector_ids = self.pinecone.upsert_vectors(successful_chunks, embeddings)
            
            if vector_ids:
                # 更新成功的新闻为completed
                successful_news_ids = list(processed_news_ids)
                # 为每个成功的新闻创建一个代表性的vector_id
                representative_vector_ids = []
                for news_id in successful_news_ids:
                    # 找到这个新闻的第一个chunk的vector_id作为代表
                    for chunk in successful_chunks:
                        if chunk['news_item']['id'] == news_id:
                            representative_vector_ids.append(chunk['chunk_id'])
                            break
                
                self.update_news_status(successful_news_ids, 'completed', representative_vector_ids)
                logger.info(f"✅ 成功处理 {len(successful_news_ids)} 篇新闻，生成 {len(vector_ids)} 个文本块向量")
            else:
                # 上传失败，恢复为pending
                self.update_news_status(news_ids, 'pending')
                logger.error("向量上传失败")
                return 0
            
            # 处理失败的新闻
            failed_news_ids = []
            for news_id in news_ids:
                if news_id not in processed_news_ids:
                    failed_news_ids.append(news_id)
            
            if failed_news_ids:
                self.update_news_status(failed_news_ids, 'failed')
                logger.warning(f"❌ {len(failed_news_ids)} 个新闻项处理失败")
            
            return len(successful_news_ids)
            
        except Exception as e:
            logger.error(f"批处理失败: {e}")
            # 恢复所有状态为pending
            self.update_news_status(news_ids, 'pending')
            return 0
    
    def run(self):
        """运行主处理流程"""
        logger.info("=" * 60)
        logger.info("开始新闻Embedding处理")
        logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        try:
            # 获取待处理的新闻
            pending_items = self.get_pending_news_items()
            
            if not pending_items:
                logger.info("没有待处理的新闻项")
                return
            
            # 估算成本
            cost_estimate = self.embedder.estimate_cost(pending_items)
            logger.info(f"成本估算: {cost_estimate}")
            
            # 分批处理
            total_processed = 0
            batches = [pending_items[i:i + self.batch_size] 
                      for i in range(0, len(pending_items), self.batch_size)]
            
            for i, batch in enumerate(batches):
                logger.info(f"\n处理批次 {i + 1}/{len(batches)}...")
                processed = self.process_batch(batch)
                total_processed += processed
                
                # 避免速率限制
                if i < len(batches) - 1:
                    import time
                    time.sleep(5)  # Gemini API速率限制
            
            # 获取统计信息
            stats = self.pinecone.get_index_stats()
            
            logger.info("\n" + "=" * 60)
            logger.info("处理完成！")
            logger.info(f"总处理数量: {total_processed}/{len(pending_items)}")
            logger.info(f"Pinecone索引统计: {stats}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"运行过程中出错: {e}", exc_info=True)
            raise

def main():
    """主函数"""
    try:
        processor = NewsEmbeddingProcessor()
        processor.run()
    except Exception as e:
        logger.error(f"程序异常退出: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()