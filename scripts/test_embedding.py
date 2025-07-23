#!/usr/bin/env python3
"""
测试Embedding流程的脚本
用于本地测试各个组件是否正常工作
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试是否能成功导入所有模块"""
    print("测试模块导入...")
    try:
        from database.supabase_handler import SupabaseHandler
        print("✅ Supabase模块导入成功")
    except Exception as e:
        print(f"❌ Supabase模块导入失败: {e}")
        return False
    
    try:
        from vector.pinecone_handler import PineconeHandler
        print("✅ Pinecone模块导入成功")
    except Exception as e:
        print(f"❌ Pinecone模块导入失败: {e}")
        return False
    
    try:
        from vector.text_embedder import TextEmbedder
        print("✅ TextEmbedder模块导入成功")
    except Exception as e:
        print(f"❌ TextEmbedder模块导入失败: {e}")
        return False
    
    return True

def test_connections():
    """测试各个服务的连接"""
    print("\n测试服务连接...")
    
    # 测试Supabase连接
    try:
        from database.supabase_handler import SupabaseHandler
        handler = SupabaseHandler()
        if handler.client:
            print("✅ Supabase连接成功")
        else:
            print("❌ Supabase连接失败")
    except Exception as e:
        print(f"❌ Supabase连接错误: {e}")
    
    # 测试Pinecone连接
    try:
        from vector.pinecone_handler import PineconeHandler
        handler = PineconeHandler()
        if handler.check_connection():
            stats = handler.get_index_stats()
            print(f"✅ Pinecone连接成功，索引统计: {stats}")
        else:
            print("❌ Pinecone连接失败")
    except Exception as e:
        print(f"❌ Pinecone连接错误: {e}")
    
    # 测试Gemini API连接
    try:
        from vector.text_embedder import TextEmbedder
        embedder = TextEmbedder()
        if embedder.check_connection():
            print("✅ Gemini API连接成功")
        else:
            print("❌ Gemini API连接失败")
    except Exception as e:
        print(f"❌ Gemini API连接错误: {e}")

def test_single_embedding():
    """测试单个文本的embedding生成"""
    print("\n测试单个文本embedding...")
    
    try:
        from vector.text_embedder import TextEmbedder
        embedder = TextEmbedder()
        
        test_text = "这是一条测试新闻：36氪报道，某科技公司获得A轮融资。该公司成立于2018年，专注于人工智能技术研发。本轮融资将主要用于技术研发和市场拓展。"
        embedding = embedder.generate_single_embedding(test_text)
        
        if embedding and len(embedding) == 768:
            print(f"✅ Embedding生成成功，维度: {len(embedding)}")
            print(f"   前5个值: {embedding[:5]}")
            
            # 测试文本分割功能
            print("\n测试文本分割功能...")
            test_news = {
                'id': 'test-123',
                'title': '测试新闻',
                'content': test_text * 5,  # 重复5次制造长文本
                'url': 'https://test.com',
                'published_at': '2024-01-01T00:00:00Z'
            }
            
            embeddings, chunks = embedder.generate_embeddings([test_news])
            print(f"✅ 文本分割测试：生成了 {len(chunks)} 个文本块")
            for i, chunk in enumerate(chunks[:3]):  # 只显示前3个
                print(f"   块{i+1}: 长度={len(chunk['text'])}, 行{chunk['line_start']}-{chunk['line_end']}")
        else:
            print("❌ Embedding生成失败")
    except Exception as e:
        print(f"❌ Embedding生成错误: {e}")
        import traceback
        traceback.print_exc()

def test_full_pipeline():
    """测试完整的处理流程（只处理1条记录）"""
    print("\n测试完整处理流程...")
    
    try:
        from scripts.embed_news import NewsEmbeddingProcessor
        
        # 创建处理器
        processor = NewsEmbeddingProcessor()
        
        # 修改配置为只处理1条
        processor.max_items_per_run = 1
        processor.batch_size = 1
        
        # 获取待处理项
        pending_items = processor.get_pending_news_items(limit=1)
        
        if pending_items:
            print(f"找到待处理项: {pending_items[0]['title'][:50]}...")
            
            # 估算成本
            cost = processor.embedder.estimate_cost(pending_items)
            print(f"成本估算: ${cost['estimated_cost_usd']}")
            
            # 处理
            processed = processor.process_batch(pending_items)
            if processed > 0:
                print("✅ 完整流程测试成功！")
            else:
                print("❌ 处理失败")
        else:
            print("⚠️  没有找到待处理的新闻项")
            print("   请先运行爬虫获取一些新闻，或检查数据库中是否有embedding_status='pending'的记录")
            
    except Exception as e:
        print(f"❌ 完整流程测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    print("=" * 60)
    print("Embedding系统测试")
    print("=" * 60)
    
    # 检查配置文件
    config_path = project_root / "vector" / "embedding_config.json"
    if not config_path.exists():
        print("❌ 配置文件不存在！")
        print(f"   请创建: {config_path}")
        print("   可以从embedding_config.json.template复制并填入API密钥")
        return
    
    # 运行测试
    if test_imports():
        test_connections()
        test_single_embedding()
        
        print("\n是否要测试完整流程？这将处理1条真实的新闻记录。")
        response = input("输入 'y' 继续，其他键跳过: ")
        if response.lower() == 'y':
            test_full_pipeline()
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()