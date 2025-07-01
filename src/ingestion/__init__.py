from typing import List, Dict, Any, Optional, Union
import os
from pathlib import Path
from datetime import datetime

from .document_parser import parse_document, DocumentParserFactory
from .chunking import get_chunker
from .vector_store import get_vector_store
from ..retrieval.embedder import get_embedder
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNKING_STRATEGY


class IngestionPipeline:
    """Main ingestion pipeline orchestrator"""
    
    def __init__(self, index_name: str = "rag-documents"):
        self.index_name = index_name
        self.vector_store = get_vector_store()
        self.embedder = get_embedder()
        self.chunker = get_chunker(
            strategy=CHUNKING_STRATEGY,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Ensure index exists
        self._ensure_index_exists()
    
    def _ensure_index_exists(self):
        """Ensure the vector store index exists"""
        try:
            existing_indexes = self.vector_store.list_indexes()
            if self.index_name not in existing_indexes:
                dimension = self.embedder.get_embedding_dimension()
                self.vector_store.create_index(self.index_name, dimension)
                print(f"Created index '{self.index_name}' with dimension {dimension}")
            else:
                print(f"Using existing index '{self.index_name}'")
        except Exception as e:
            print(f"Warning: Could not verify index existence: {str(e)}")
    
    def ingest_document(self, file_path_or_bytes: Union[str, Path, bytes], 
                       file_extension: str = None, 
                       additional_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Ingest a single document through the complete pipeline
        
        Args:
            file_path_or_bytes: File path or bytes content
            file_extension: File extension (required for bytes input)
            additional_metadata: Additional metadata to add to all chunks
        
        Returns:
            Dict with ingestion results and statistics
        """
        try:
            start_time = datetime.now()
            
            # Step 1: Parse document
            print(f"🔍 Parsing document...")
            parsed_doc = parse_document(file_path_or_bytes, file_extension)
            
            # Add additional metadata if provided
            if additional_metadata:
                parsed_doc['metadata'].update(additional_metadata)
            
            # Add ingestion timestamp
            parsed_doc['metadata']['ingested_at'] = start_time.isoformat()
            
            # print(f"🔍 Parsed document: {parsed_doc}")
            
            # Step 2: Chunk the document
            print(f"✂️  Chunking document...")
            chunks = self.chunker.chunk_text(
                text=parsed_doc['text'],
                metadata=parsed_doc['metadata']
            )
            
            # Step 3: Add documents to vector store
            print(f"🔢 Embedding and storing {len(chunks)} chunks...")
            self.vector_store.add_documents(
                index_name=self.index_name,
                documents=chunks,
                embedder=self.embedder
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Return results
            result = {
                'success': True,
                'filename': parsed_doc['metadata'].get('filename', 'unknown'),
                'file_type': parsed_doc['metadata'].get('file_type', 'unknown'),
                'total_chunks': len(chunks),
                'processing_time_seconds': processing_time,
                'index_name': self.index_name,
                'metadata': parsed_doc['metadata']
            }
            
            print(f"✅ Successfully ingested document in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'filename': getattr(file_path_or_bytes, 'name', 'unknown'),
                'processing_time_seconds': (datetime.now() - start_time).total_seconds()
            }
            print(f"❌ Failed to ingest document: {str(e)}")
            return error_result
    
    def ingest_documents_batch(self, file_paths: List[Union[str, Path]], 
                              additional_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Ingest multiple documents in batch
        
        Args:
            file_paths: List of file paths to ingest
            additional_metadata: Additional metadata to add to all chunks
        
        Returns:
            List of ingestion results for each document
        """
        results = []
        total_files = len(file_paths)
        
        print(f"📚 Starting batch ingestion of {total_files} documents...")
        
        for i, file_path in enumerate(file_paths, 1):
            print(f"\n📄 Processing file {i}/{total_files}: {file_path}")
            
            # Add batch metadata
            batch_metadata = {
                'batch_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'batch_position': i,
                'batch_total': total_files
            }
            
            if additional_metadata:
                batch_metadata.update(additional_metadata)
            
            result = self.ingest_document(file_path, additional_metadata=batch_metadata)
            results.append(result)
        
        # Summary statistics
        successful = sum(1 for r in results if r['success'])
        failed = total_files - successful
        total_chunks = sum(r.get('total_chunks', 0) for r in results if r['success'])
        total_time = sum(r.get('processing_time_seconds', 0) for r in results)
        
        print(f"\n📊 Batch ingestion complete:")
        print(f"  ✅ Successful: {successful}/{total_files}")
        print(f"  ❌ Failed: {failed}/{total_files}")
        print(f"  📝 Total chunks: {total_chunks}")
        print(f"  ⏱️  Total time: {total_time:.2f} seconds")
        
        return results
    
    def ingest_directory(self, directory_path: Union[str, Path], 
                        recursive: bool = True,
                        additional_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Ingest all supported documents from a directory
        
        Args:
            directory_path: Path to directory containing documents
            recursive: Whether to search subdirectories
            additional_metadata: Additional metadata to add to all chunks
        
        Returns:
            List of ingestion results for each document
        """
        directory_path = Path(directory_path)
        supported_extensions = DocumentParserFactory.supported_extensions()
        
        # Find all supported files
        file_paths = []
        if recursive:
            for ext in supported_extensions:
                file_paths.extend(directory_path.rglob(f"*{ext}"))
        else:
            for ext in supported_extensions:
                file_paths.extend(directory_path.glob(f"*{ext}"))
        
        if not file_paths:
            print(f"⚠️  No supported documents found in {directory_path}")
            return []
        
        print(f"🔍 Found {len(file_paths)} supported documents in {directory_path}")
        
        # Add directory metadata
        dir_metadata = {
            'source_directory': str(directory_path),
            'recursive_search': recursive
        }
        
        if additional_metadata:
            dir_metadata.update(additional_metadata)
        
        return self.ingest_documents_batch(file_paths, dir_metadata)
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        try:
            # This is a basic implementation - you might want to add more detailed stats
            return {
                'index_name': self.index_name,
                'vector_store_type': type(self.vector_store).__name__,
                'embedder_type': type(self.embedder).__name__,
                'chunking_strategy': CHUNKING_STRATEGY,
                'chunk_size': CHUNK_SIZE,
                'chunk_overlap': CHUNK_OVERLAP
            }
        except Exception as e:
            return {'error': str(e)}


def create_ingestion_pipeline(index_name: str = "rag-documents") -> IngestionPipeline:
    """Factory function to create an ingestion pipeline"""
    return IngestionPipeline(index_name)


# Convenience functions for direct use
def ingest_single_document(file_path_or_bytes: Union[str, Path, bytes], 
                          file_extension: str = None,
                          index_name: str = "rag-documents",
                          additional_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenience function to ingest a single document"""
    pipeline = create_ingestion_pipeline(index_name)
    return pipeline.ingest_document(file_path_or_bytes, file_extension, additional_metadata)


def ingest_directory(directory_path: Union[str, Path],
                    index_name: str = "rag-documents",
                    recursive: bool = True,
                    additional_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Convenience function to ingest a directory of documents"""
    pipeline = create_ingestion_pipeline(index_name)
    return pipeline.ingest_directory(directory_path, recursive, additional_metadata)
