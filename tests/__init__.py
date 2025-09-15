"""
SecureChat Analytics Platform - Test Suite

This package contains comprehensive tests for the SecureChat Analytics Platform,
including unit tests, integration tests, security tests, and performance benchmarks.

Test Categories:
- Unit Tests: Core functionality and component testing
- Integration Tests: Gradio interface and workflow testing  
- Security Tests: Input validation and privacy protection
- Performance Tests: Response time and resource usage benchmarks

Usage:
    python -m pytest tests/ -v
    python -m pytest tests/ -v -k "security"
    python -m pytest tests/ -v --cov=app_local --cov-report=html
"""

__version__ = "1.0.0"
__author__ = "SecureChat Analytics Team"

# Test configuration constants
TEST_CONFIG = {
    "MAX_RESPONSE_TIME": 5.0,  # Maximum acceptable response time in seconds
    "MAX_MEMORY_USAGE": 4096,  # Maximum memory usage in MB
    "MIN_SANITIZATION_SPEED": 1.0,  # Minimum sanitization speed requirement
    "TEST_TIMEOUT": 30,  # Test timeout in seconds
}

# Test data samples
SAMPLE_INPUTS = {
    "clean_text": "Hello, this is a normal business message.",
    "malicious_html": "<script>alert('xss')</script>Hello World!",
    "sql_injection": "'; DROP TABLE users; --",
    "pii_text": "My name is John Smith and my email is john.smith@example.com",
    "long_text": "This is a very long text " * 100,
    "empty_text": "",
    "special_chars": "Hello@#$%^&*()World!",
}

EXPECTED_OUTPUTS = {
    "sanitized_html": "Hello World!",
    "sanitized_sql": " DROP TABLE users ",
    "sanitized_special": "HelloWorld!",
}

# Mock responses for testing
MOCK_RESPONSES = {
    "sentiment_analysis": "Sentiment: Positive\nTone: Friendly\nSecurity Risk: Low",
    "secure_summary": "A business communication regarding general topics.",
    "content_classification": "Content Type: Business\nSafety Level: Safe",
}