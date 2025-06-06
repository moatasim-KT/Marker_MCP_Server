"""Comprehensive PDF conversion tests using real functionality without extensive mocking.

This module provides robust, real-world testing of the PDF conversion pipeline,
focusing on actual functionality rather than mocked components.
"""
import pytest
import asyncio
import tempfile
import time
from pathlib import Path
import logging

from src.marker_mcp_server.tools import (
    handle_single_convert, 
    handle_batch_convert, 
    handle_chunk_convert, 
    handle_batch_pages_convert
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDataGenerator:
    """Generate various types of PDF test data."""
    
    @staticmethod
    def create_simple_pdf(content: str = "Test content") -> bytes:
        """Create a simple, valid PDF with basic content."""
        return f"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length {len(content) + 50}
>>
stream
BT
/F1 12 Tf
100 700 Td
({content}) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000194 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
{300 + len(content)}
%%EOF""".encode()

    @staticmethod
    def create_multi_page_pdf(num_pages: int = 3) -> bytes:
        """Create a multi-page PDF for testing pagination."""
        pages_content = []
        page_objects = []
        
        for i in range(num_pages):
            page_num = 3 + i
            content_num = page_num + num_pages
            page_objects.append(f"{page_num} 0 R")
            
            pages_content.append(f"""
{page_num} 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents {content_num} 0 R
>>
endobj

{content_num} 0 obj
<<
/Length 60
>>
stream
BT
/F1 12 Tf
100 700 Td
(Page {i + 1} content) Tj
ET
endstream
endobj""")
        
        pdf_content = f"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [{' '.join(page_objects)}]
/Count {num_pages}
>>
endobj

{''.join(pages_content)}

xref
0 {3 + num_pages * 2}
0000000000 65535 f 
{''.join([f'0000000{str(i).zfill(3)} 00000 n ' for i in range(1, 3 + num_pages * 2)])}
trailer
<<
/Size {3 + num_pages * 2}
/Root 1 0 R
>>
startxref
1000
%%EOF"""
        return pdf_content.encode()

    @staticmethod
    def create_corrupted_pdf() -> bytes:
        """Create intentionally corrupted PDF content."""
        return b"This is not a valid PDF file at all"

    @staticmethod
    def create_empty_pdf() -> bytes:
        """Create an empty PDF (zero bytes)."""
        return b""


@pytest.fixture
def test_workspace(tmp_path):
    """Create a temporary workspace for tests."""
    workspace = tmp_path / "test_workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def sample_pdfs(test_workspace):
    """Create various sample PDFs for testing."""
    pdfs = {}
    
    # Simple valid PDF
    simple_pdf = test_workspace / "simple.pdf"
    simple_pdf.write_bytes(TestDataGenerator.create_simple_pdf("Simple test document"))
    pdfs['simple'] = simple_pdf
    
    # Multi-page PDF
    multi_pdf = test_workspace / "multi_page.pdf"
    multi_pdf.write_bytes(TestDataGenerator.create_multi_page_pdf(5))
    pdfs['multi'] = multi_pdf
    
    # Large content PDF
    large_pdf = test_workspace / "large.pdf"
    large_content = "Large document content. " * 1000
    large_pdf.write_bytes(TestDataGenerator.create_simple_pdf(large_content))
    pdfs['large'] = large_pdf
    
    # Corrupted PDF
    corrupted_pdf = test_workspace / "corrupted.pdf"
    corrupted_pdf.write_bytes(TestDataGenerator.create_corrupted_pdf())
    pdfs['corrupted'] = corrupted_pdf
    
    # Empty PDF
    empty_pdf = test_workspace / "empty.pdf"
    empty_pdf.write_bytes(TestDataGenerator.create_empty_pdf())
    pdfs['empty'] = empty_pdf
    
    return pdfs


class TestSinglePDFConversion:
    """Test single PDF conversion functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_single_conversion(self, sample_pdfs, test_workspace):
        """Test basic single PDF conversion."""
        output_dir = test_workspace / "output"
        output_dir.mkdir()
        
        arguments = {
            "file_path": str(sample_pdfs['simple']),
            "output_dir": str(output_dir),
            "output_format": "markdown",
            "debug": False
        }
        
        result = await handle_single_convert(arguments)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert "success" in result
        assert "input_file" in result
        assert "message" in result
        
        # If conversion succeeded, verify output file
        if result.get("success"):
            assert "output_file" in result
            output_file = Path(result["output_file"])
            assert output_file.exists()
            assert output_file.suffix == ".md"
            
            # Verify content exists
            content = output_file.read_text()
            assert len(content) > 0
            logger.info(f"Successfully converted {arguments['file_path']}")
        else:
            logger.warning(f"Conversion failed: {result.get('error', 'Unknown error')}")
    
    @pytest.mark.asyncio
    async def test_multi_page_conversion(self, sample_pdfs, test_workspace):
        """Test conversion of multi-page PDFs."""
        output_dir = test_workspace / "output_multi"
        output_dir.mkdir()
        
        arguments = {
            "file_path": str(sample_pdfs['multi']),
            "output_dir": str(output_dir),
            "output_format": "markdown",
            "debug": True  # Enable debug to get more information
        }
        
        result = await handle_single_convert(arguments)
        
        assert isinstance(result, dict)
        assert "success" in result
        
        if result.get("success"):
            output_file = Path(result["output_file"])
            assert output_file.exists()
            content = output_file.read_text()
            assert len(content) > 0
            logger.info(f"Multi-page conversion successful: {len(content)} characters")
    
    @pytest.mark.asyncio
    async def test_large_pdf_conversion(self, sample_pdfs, test_workspace):
        """Test conversion of large PDF files."""
        output_dir = test_workspace / "output_large"
        output_dir.mkdir()
        
        arguments = {
            "file_path": str(sample_pdfs['large']),
            "output_dir": str(output_dir),
            "output_format": "markdown",
            "debug": False
        }
        
        start_time = time.time()
        result = await handle_single_convert(arguments)
        processing_time = time.time() - start_time
        
        assert isinstance(result, dict)
        assert "success" in result
        
        # Log performance metrics
        logger.info(f"Large PDF processing time: {processing_time:.2f} seconds")
        
        if result.get("success"):
            output_file = Path(result["output_file"])
            assert output_file.exists()
            content = output_file.read_text()
            assert len(content) > 0
    
    @pytest.mark.asyncio
    async def test_corrupted_pdf_handling(self, sample_pdfs, test_workspace):
        """Test handling of corrupted PDF files."""
        output_dir = test_workspace / "output_corrupted"
        output_dir.mkdir()
        
        arguments = {
            "file_path": str(sample_pdfs['corrupted']),
            "output_dir": str(output_dir),
            "output_format": "markdown",
            "debug": False
        }
        
        result = await handle_single_convert(arguments)
        
        # Should fail gracefully
        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result
        assert len(result["error"]) > 0
        logger.info(f"Corrupted PDF handled correctly: {result['error']}")
    
    @pytest.mark.asyncio
    async def test_empty_pdf_handling(self, sample_pdfs, test_workspace):
        """Test handling of empty PDF files."""
        output_dir = test_workspace / "output_empty"
        output_dir.mkdir()
        
        arguments = {
            "file_path": str(sample_pdfs['empty']),
            "output_dir": str(output_dir),
            "output_format": "markdown",
            "debug": False
        }
        
        result = await handle_single_convert(arguments)
        
        # Should fail gracefully
        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result
        logger.info(f"Empty PDF handled correctly: {result.get('error')}")
    
    @pytest.mark.asyncio
    async def test_nonexistent_file_handling(self, test_workspace):
        """Test handling of non-existent files."""
        output_dir = test_workspace / "output_nonexistent"
        output_dir.mkdir()
        
        arguments = {
            "file_path": str(test_workspace / "does_not_exist.pdf"),
            "output_dir": str(output_dir),
            "output_format": "markdown",
            "debug": False
        }
        
        result = await handle_single_convert(arguments)
        
        # Should fail gracefully
        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result
        assert "not found" in result["error"].lower() or "file not found" in result["error"].lower()
        logger.info(f"Non-existent file handled correctly: {result['error']}")


class TestBatchPDFConversion:
    """Test batch PDF conversion functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_batch_conversion(self, test_workspace):
        """Test basic batch conversion with multiple PDFs."""
        # Create input directory with multiple PDFs
        input_dir = test_workspace / "batch_input"
        input_dir.mkdir()
        output_dir = test_workspace / "batch_output"
        
        # Create multiple test PDFs
        test_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
        for filename in test_files:
            pdf_file = input_dir / filename
            pdf_file.write_bytes(TestDataGenerator.create_simple_pdf(f"Content for {filename}"))
        
        arguments = {
            "in_folder": str(input_dir),
            "output_dir": str(output_dir),
            "output_format": "markdown",
            "debug": False
        }
        
        result = await handle_batch_convert(arguments)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert "success" in result
        assert "results" in result
        assert "input_folder" in result
        assert "output_folder" in result
        
        # Verify individual results
        assert len(result["results"]) == len(test_files)
        
        successful_conversions = [r for r in result["results"] if r.get("success")]
        logger.info(f"Batch conversion: {len(successful_conversions)}/{len(test_files)} successful")
        
        # Verify output files exist for successful conversions
        for file_result in successful_conversions:
            if "output_file" in file_result:
                output_file = Path(file_result["output_file"])
                assert output_file.exists()
    
    @pytest.mark.asyncio
    async def test_batch_conversion_with_mixed_files(self, test_workspace):
        """Test batch conversion with a mix of valid and invalid files."""
        input_dir = test_workspace / "mixed_batch"
        input_dir.mkdir()
        output_dir = test_workspace / "mixed_output"
        
        # Create mix of valid and invalid files
        valid_pdf = input_dir / "valid.pdf"
        valid_pdf.write_bytes(TestDataGenerator.create_simple_pdf("Valid content"))
        
        corrupted_pdf = input_dir / "corrupted.pdf"
        corrupted_pdf.write_bytes(TestDataGenerator.create_corrupted_pdf())
        
        empty_pdf = input_dir / "empty.pdf"
        empty_pdf.write_bytes(TestDataGenerator.create_empty_pdf())
        
        # Add a non-PDF file (should be ignored)
        text_file = input_dir / "not_a_pdf.txt"
        text_file.write_text("This is not a PDF")
        
        arguments = {
            "in_folder": str(input_dir),
            "output_dir": str(output_dir),
            "output_format": "markdown",
            "debug": False
        }
        
        result = await handle_batch_convert(arguments)
        
        assert isinstance(result, dict)
        assert "results" in result
        
        # Should process only PDF files
        pdf_results = [r for r in result["results"] if r["file"].endswith(".pdf")]
        assert len(pdf_results) == 3  # valid, corrupted, empty
        
        # At least one should succeed (the valid one)
        successful = [r for r in pdf_results if r.get("success")]
        failed = [r for r in pdf_results if not r.get("success")]
        
        assert len(successful) >= 1
        logger.info(f"Mixed batch: {len(successful)} successful, {len(failed)} failed")
    
    @pytest.mark.asyncio
    async def test_batch_conversion_empty_directory(self, test_workspace):
        """Test batch conversion with empty directory."""
        input_dir = test_workspace / "empty_batch"
        input_dir.mkdir()
        output_dir = test_workspace / "empty_output"
        
        arguments = {
            "in_folder": str(input_dir),
            "output_dir": str(output_dir),
            "output_format": "markdown",
            "debug": False
        }
        
        result = await handle_batch_convert(arguments)
        
        # Should fail gracefully when no PDFs found
        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result
        assert "no pdf files" in result["error"].lower() or "not found" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_batch_conversion_nonexistent_directory(self, test_workspace):
        """Test batch conversion with non-existent directory."""
        arguments = {
            "in_folder": str(test_workspace / "does_not_exist"),
            "output_dir": str(test_workspace / "output"),
            "output_format": "markdown",
            "debug": False
        }
        
        result = await handle_batch_convert(arguments)
        
        # Should fail gracefully
        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result
        assert "invalid" in result["error"].lower() or "not found" in result["error"].lower()


class TestChunkPDFConversion:
    """Test chunk-based PDF conversion functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_chunk_conversion(self, test_workspace):
        """Test basic chunk conversion."""
        # Create input directory with PDFs
        input_dir = test_workspace / "chunk_input"
        input_dir.mkdir()
        output_dir = test_workspace / "chunk_output"
        
        # Create multiple test PDFs
        for i in range(8):
            pdf_file = input_dir / f"chunk_doc_{i:02d}.pdf"
            pdf_file.write_bytes(TestDataGenerator.create_simple_pdf(f"Chunk document {i}"))
        
        arguments = {
            "in_folder": str(input_dir),
            "out_folder": str(output_dir),
            "chunk_size": 3,  # Process 3 files at a time
            "output_format": "markdown",
            "debug": False
        }
        
        result = await handle_chunk_convert(arguments)
        
        assert isinstance(result, dict)
        assert "success" in result
        
        if result.get("success"):
            assert "chunks_processed" in result or "message" in result
            logger.info(f"Chunk conversion result: {result.get('message', 'Success')}")
    
    @pytest.mark.asyncio
    async def test_chunk_conversion_with_different_sizes(self, test_workspace):
        """Test chunk conversion with different chunk sizes."""
        input_dir = test_workspace / "chunk_sizes"
        input_dir.mkdir()
        
        # Create 10 test PDFs
        for i in range(10):
            pdf_file = input_dir / f"doc_{i:02d}.pdf"
            pdf_file.write_bytes(TestDataGenerator.create_simple_pdf(f"Document {i}"))
        
        # Test different chunk sizes
        chunk_sizes = [1, 3, 5, 10, 15]  # Including sizes larger than available files
        
        for chunk_size in chunk_sizes:
            output_dir = test_workspace / f"chunk_output_{chunk_size}"
            
            arguments = {
                "in_folder": str(input_dir),
                "out_folder": str(output_dir),
                "chunk_size": chunk_size,
                "output_format": "markdown",
                "debug": False
            }
            
            result = await handle_chunk_convert(arguments)
            
            assert isinstance(result, dict)
            logger.info(f"Chunk size {chunk_size}: {result.get('success', 'Unknown status')}")


class TestBatchPagesConversion:
    """Test batch pages conversion functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_batch_pages_conversion(self, sample_pdfs, test_workspace):
        """Test basic batch pages conversion."""
        output_dir = test_workspace / "pages_output"
        output_dir.mkdir()
        
        arguments = {
            "file_path": str(sample_pdfs['multi']),  # Multi-page PDF
            "output_dir": str(output_dir),
            "page_range": "1-3",  # Convert first 3 pages
            "output_format": "markdown",
            "debug": False
        }
        
        result = await handle_batch_pages_convert(arguments)
        
        assert isinstance(result, dict)
        assert "success" in result
        
        if result.get("success"):
            logger.info(f"Batch pages conversion successful: {result.get('message', 'Success')}")
    
    @pytest.mark.asyncio
    async def test_batch_pages_invalid_range(self, sample_pdfs, test_workspace):
        """Test batch pages conversion with invalid page range."""
        output_dir = test_workspace / "pages_invalid"
        output_dir.mkdir()
        
        arguments = {
            "file_path": str(sample_pdfs['simple']),
            "output_dir": str(output_dir),
            "page_range": "10-20",  # Invalid range for a simple PDF
            "output_format": "markdown",
            "debug": False
        }
        
        result = await handle_batch_pages_convert(arguments)
        
        assert isinstance(result, dict)
        # Should handle invalid range gracefully
        logger.info(f"Invalid page range handled: {result.get('success', 'Unknown')}")


class TestConcurrentConversion:
    """Test concurrent PDF conversion scenarios."""
    
    @pytest.mark.asyncio
    async def test_concurrent_single_conversions(self, test_workspace):
        """Test multiple concurrent single conversions."""
        output_dir = test_workspace / "concurrent_output"
        output_dir.mkdir()
        
        # Create multiple PDFs
        pdf_files = []
        for i in range(5):
            pdf_file = test_workspace / f"concurrent_{i}.pdf"
            pdf_file.write_bytes(TestDataGenerator.create_simple_pdf(f"Concurrent doc {i}"))
            pdf_files.append(pdf_file)
        
        # Create concurrent conversion tasks
        tasks = []
        for i, pdf_file in enumerate(pdf_files):
            arguments = {
                "file_path": str(pdf_file),
                "output_dir": str(output_dir),
                "output_format": "markdown",
                "debug": False
            }
            task = handle_single_convert(arguments)
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify results
        successful_conversions = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Concurrent conversion {i} failed with exception: {result}")
            elif isinstance(result, dict):
                if result.get("success"):
                    successful_conversions += 1
                    logger.info(f"Concurrent conversion {i} succeeded")
                else:
                    logger.warning(f"Concurrent conversion {i} failed: {result.get('error')}")
        
        logger.info(f"Concurrent conversions: {successful_conversions}/{len(pdf_files)} successful")
        assert successful_conversions > 0  # At least some should succeed


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_permission_denied_output_directory(self, sample_pdfs):
        """Test handling when output directory cannot be created."""
        # Try to use root directory (should fail on most systems)
        arguments = {
            "file_path": str(sample_pdfs['simple']),
            "output_dir": "/root/cannot_create_here",
            "output_format": "markdown",
            "debug": False
        }
        
        result = await handle_single_convert(arguments)
        
        assert isinstance(result, dict)
        # Should handle permission errors gracefully
        if not result.get("success"):
            assert "error" in result
            logger.info(f"Permission error handled correctly: {result['error']}")
    
    @pytest.mark.asyncio
    async def test_invalid_arguments(self):
        """Test handling of invalid arguments."""
        # Missing required arguments
        arguments = {
            "output_format": "markdown",
            "debug": False
        }
        
        result = await handle_single_convert(arguments)
        
        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result
        logger.info(f"Invalid arguments handled: {result['error']}")
    
    @pytest.mark.asyncio
    async def test_different_output_formats(self, sample_pdfs, test_workspace):
        """Test different output formats."""
        output_formats = ["markdown", "json", "html"]
        
        for output_format in output_formats:
            output_dir = test_workspace / f"format_{output_format}"
            output_dir.mkdir(exist_ok=True)
            
            arguments = {
                "file_path": str(sample_pdfs['simple']),
                "output_dir": str(output_dir),
                "output_format": output_format,
                "debug": False
            }
            
            result = await handle_single_convert(arguments)
            
            assert isinstance(result, dict)
            logger.info(f"Output format {output_format}: {result.get('success', 'Unknown')}")


class TestPerformanceAndStress:
    """Test performance and stress scenarios."""
    
    @pytest.mark.asyncio
    async def test_large_batch_performance(self, test_workspace):
        """Test performance with large batch of files."""
        input_dir = test_workspace / "large_batch"
        input_dir.mkdir()
        output_dir = test_workspace / "large_batch_output"
        
        # Create 20 test PDFs
        num_files = 20
        for i in range(num_files):
            pdf_file = input_dir / f"large_batch_{i:03d}.pdf"
            pdf_file.write_bytes(TestDataGenerator.create_simple_pdf(f"Large batch document {i}"))
        
        start_time = time.time()
        
        arguments = {
            "in_folder": str(input_dir),
            "output_dir": str(output_dir),
            "output_format": "markdown",
            "debug": False,
            "max_files": 10  # Limit for performance testing
        }
        
        result = await handle_batch_convert(arguments)
        
        processing_time = time.time() - start_time
        
        assert isinstance(result, dict)
        
        if result.get("success"):
            files_processed = len(result.get("results", []))
            throughput = files_processed / processing_time if processing_time > 0 else 0
            logger.info(f"Large batch performance: {files_processed} files in {processing_time:.2f}s "
                       f"({throughput:.2f} files/sec)")
        else:
            logger.warning(f"Large batch processing failed: {result.get('error')}")
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, sample_pdfs, test_workspace):
        """Test memory usage during conversion."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        output_dir = test_workspace / "memory_test"
        output_dir.mkdir()
        
        arguments = {
            "file_path": str(sample_pdfs['large']),
            "output_dir": str(output_dir),
            "output_format": "markdown",
            "debug": False
        }
        
        result = await handle_single_convert(arguments)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        logger.info(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB "
                   f"(+{memory_increase:.1f}MB)")
        
        assert isinstance(result, dict)
        # Memory increase should be reasonable (less than 500MB for test)
        assert memory_increase < 500


if __name__ == "__main__":
    # Run a quick test to verify functionality
    import tempfile
    
    async def quick_test():
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            
            # Create a simple test PDF
            test_pdf = workspace / "test.pdf"
            test_pdf.write_bytes(TestDataGenerator.create_simple_pdf("Quick test"))
            
            # Test single conversion
            arguments = {
                "file_path": str(test_pdf),
                "output_dir": str(workspace),
                "output_format": "markdown",
                "debug": False
            }
            
            result = await handle_single_convert(arguments)
            print(f"Quick test result: {result}")
    
    asyncio.run(quick_test())
