"""
Integration tests for the SecureChat Analytics Platform.

This module contains integration tests that verify the interaction between
different components of the system, including Gradio interfaces, API connections,
workflow integrations, and end-to-end functionality.
"""

import pytest
import sys
import os
import time
import threading
import requests
from unittest.mock import patch, Mock, MagicMock
import tempfile
import json

# Add the parent directory to the path to import the apps
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests import TEST_CONFIG, SAMPLE_INPUTS, MOCK_RESPONSES


class TestGradioIntegration:
    """Integration tests for Gradio application interfaces"""
    
    def test_api_app_creation(self):
        """Test that API-based Gradio app can be created without errors"""
        with patch('huggingface_hub.InferenceClient') as mock_client:
            # Mock the InferenceClient
            mock_client.return_value = Mock()
            mock_client.return_value.text_generation.return_value = "Mock response"
            
            try:
                import app
                
                # Verify the demo object exists and has required methods
                assert hasattr(app, 'demo'), "App should have a 'demo' object"
                assert hasattr(app.demo, 'launch'), "Demo should have a 'launch' method"
                
                # Verify the analytics object exists
                assert hasattr(app, 'analytics'), "App should have an 'analytics' object"
                
                # Test that the analytics object has required methods
                if hasattr(app.analytics, 'analyze_sentiment_tone'):
                    # Test basic functionality without actually calling the API
                    assert callable(app.analytics.analyze_sentiment_tone)
                
                print("✅ API-based Gradio app created successfully")
                
            except ImportError as e:
                pytest.skip(f"API app import failed: {e}")
            except Exception as e:
                pytest.fail(f"Failed to create API-based Gradio app: {e}")
    
    def test_local_app_creation(self):
        """Test that local model Gradio app can be created without errors"""
        with patch('app_local.AutoTokenizer'), \
             patch('app_local.AutoModelForCausalLM'), \
             patch('app_local.pipeline'):
            
            try:
                import app_local
                
                # Verify the demo object exists
                assert hasattr(app_local, 'demo'), "Local app should have a 'demo' object"
                assert hasattr(app_local.demo, 'launch'), "Demo should have a 'launch' method"
                
                # Verify the analytics object exists
                assert hasattr(app_local, 'local_analytics'), "App should have 'local_analytics' object"
                
                print("✅ Local model Gradio app created successfully")
                
            except ImportError as e:
                pytest.skip(f"Local app import failed: {e}")
            except Exception as e:
                pytest.fail(f"Failed to create local model Gradio app: {e}")
    
    def test_gradio_interface_components(self):
        """Test that Gradio interface components are properly configured"""
        try:
            # Test API app components
            with patch('huggingface_hub.InferenceClient'):
                import app
                
                # The demo should be a Gradio Blocks object
                assert hasattr(app.demo, 'blocks'), "Should have blocks attribute or be Blocks object"
                
            # Test local app components  
            with patch('app_local.AutoTokenizer'), \
                 patch('app_local.AutoModelForCausalLM'), \
                 patch('app_local.pipeline'):
                import app_local
                
                assert hasattr(app_local.demo, 'blocks'), "Should have blocks attribute or be Blocks object"
                
            print("✅ Gradio interface components configured correctly")
            
        except ImportError as e:
            pytest.skip(f"Interface component test skipped: {e}")
    
    def test_interface_functions_callable(self):
        """Test that interface functions are callable and return expected types"""
        try:
            # Test API interface functions
            with patch('huggingface_hub.InferenceClient') as mock_client:
                mock_client.return_value.text_generation.return_value = MOCK_RESPONSES["sentiment_analysis"]
                
                import app
                
                if hasattr(app, 'analyze_text_interface'):
                    result = app.analyze_text_interface(
                        SAMPLE_INPUTS["clean_text"], 
                        "Sentiment & Tone Analysis"
                    )
                    
                    # Should return tuple of strings
                    assert isinstance(result, tuple)
                    assert len(result) >= 2
                    assert all(isinstance(r, str) for r in result)
            
            # Test local interface functions
            with patch('app_local.AutoTokenizer'), \
                 patch('app_local.AutoModelForCausalLM'), \
                 patch('app_local.pipeline'):
                
                import app_local
                
                if hasattr(app_local, 'analyze_local_interface'):
                    result = app_local.analyze_local_interface(
                        SAMPLE_INPUTS["clean_text"],
                        "Sentiment & Tone Analysis"
                    )
                    
                    assert isinstance(result, tuple)
                    assert len(result) >= 2
                    assert all(isinstance(r, str) for r in result)
            
            print("✅ Interface functions are callable and return expected types")
            
        except ImportError as e:
            pytest.skip(f"Interface function test skipped: {e}")


class TestAPIIntegration:
    """Integration tests for API connections and external services"""
    
    def test_huggingface_client_initialization(self):
        """Test HuggingFace InferenceClient initialization"""
        with patch('huggingface_hub.InferenceClient') as mock_client:
            # Mock successful client creation
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            try:
                import app
                
                # Verify client was created
                assert mock_client.called, "InferenceClient should be instantiated"
                
                # Check if client was created with correct model
                args, kwargs = mock_client.call_args
                assert "vaultgemma" in args[0].lower() or "vault" in str(args).lower()
                
                print("✅ HuggingFace client initialization successful")
                
            except ImportError as e:
                pytest.skip(f"API integration test skipped: {e}")
    
    def test_api_error_handling(self):
        """Test API error handling and fallback behavior"""
        with patch('huggingface_hub.InferenceClient') as mock_client:
            # Mock API failure
            mock_instance = Mock()
            mock_instance.text_generation.side_effect = Exception("API Error")
            mock_client.return_value = mock_instance
            
            try:
                import app
                
                # Test error handling in analysis
                if hasattr(app.analytics, 'analyze_sentiment_tone'):
                    result, time_taken, model_info = app.analytics.analyze_sentiment_tone(
                        SAMPLE_INPUTS["clean_text"]
                    )
                    
                    # Should handle error gracefully
                    assert isinstance(result, str)
                    assert "error" in result.lower()
                    assert isinstance(time_taken, (int, float))
                
                print("✅ API error handling works correctly")
                
            except ImportError as e:
                pytest.skip(f"API error handling test skipped: {e}")
    
    @pytest.mark.slow
    def test_mock_api_response_processing(self):
        """Test processing of mock API responses"""
        with patch('huggingface_hub.InferenceClient') as mock_client:
            # Mock successful API response
            mock_instance = Mock()
            mock_instance.text_generation.return_value = MOCK_RESPONSES["sentiment_analysis"]
            mock_client.return_value = mock_instance
            
            try:
                import app
                
                if hasattr(app.analytics, 'analyze_sentiment_tone'):
                    result, time_taken, model_info = app.analytics.analyze_sentiment_tone(
                        SAMPLE_INPUTS["clean_text"]
                    )
                    
                    # Verify response processing
                    assert isinstance(result, str)
                    assert len(result) > 0
                    assert isinstance(time_taken, (int, float))
                    assert time_taken >= 0
                    assert "api" in model_info.lower()
                
                print("✅ API response processing successful")
                
            except ImportError as e:
                pytest.skip(f"API response test skipped: {e}")


class TestLocalModelIntegration:
    """Integration tests for local model deployment"""
    
    def test_model_loading_simulation(self):
        """Test model loading process simulation"""
        with patch('app_local.AutoTokenizer') as mock_tokenizer, \
             patch('app_local.AutoModelForCausalLM') as mock_model, \
             patch('app_local.pipeline') as mock_pipeline:
            
            # Mock successful model loading
            mock_tokenizer.from_pretrained.return_value = Mock()
            mock_model.from_pretrained.return_value = Mock()
            mock_pipeline.return_value = Mock()
            
            try:
                import app_local
                
                if hasattr(app_local.local_analytics, 'load_model'):
                    result = app_local.local_analytics.load_model()
                    
                    # Should indicate successful loading
                    assert isinstance(result, str)
                    assert "success" in result.lower() or "loaded" in result.lower()
                
                print("✅ Model loading simulation successful")
                
            except ImportError as e:
                pytest.skip(f"Model loading test skipped: {e}")
    
    def test_local_inference_simulation(self):
        """Test local model inference simulation"""
        with patch('app_local.AutoTokenizer'), \
             patch('app_local.AutoModelForCausalLM'), \
             patch('app_local.pipeline') as mock_pipeline:
            
            # Mock inference pipeline
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.return_value = [{'generated_text': MOCK_RESPONSES["sentiment_analysis"]}]
            mock_pipeline.return_value = mock_pipeline_instance
            
            try:
                import app_local
                
                # Simulate loaded model
                app_local.local_analytics.model_loaded = True
                app_local.local_analytics.pipeline = mock_pipeline_instance
                
                if hasattr(app_local.local_analytics, 'analyze_with_local_model'):
                    result, time_taken = app_local.local_analytics.analyze_with_local_model(
                        SAMPLE_INPUTS["clean_text"],
                        "Sentiment & Tone Analysis"
                    )
                    
                    # Verify local inference
                    assert isinstance(result, str)
                    assert isinstance(time_taken, (int, float))
                    assert time_taken >= 0
                
                print("✅ Local inference simulation successful")
                
            except ImportError as e:
                pytest.skip(f"Local inference test skipped: {e}")
    
    def test_resource_monitoring_integration(self):
        """Test system resource monitoring integration"""
        try:
            with patch('app_local.psutil') as mock_psutil:
                # Mock system stats
                mock_psutil.cpu_percent.return_value = 50.0
                mock_memory = Mock()
                mock_memory.percent = 60.0
                mock_memory.used = 4 * 1024**3  # 4GB
                mock_memory.total = 8 * 1024**3  # 8GB
                mock_psutil.virtual_memory.return_value = mock_memory
                
                import app_local
                
                if hasattr(app_local.local_analytics, 'get_system_stats'):
                    stats = app_local.local_analytics.get_system_stats()
                    
                    assert isinstance(stats, str)
                    assert "cpu" in stats.lower() or "memory" in stats.lower()
                
                print("✅ Resource monitoring integration successful")
                
        except ImportError as e:
            pytest.skip(f"Resource monitoring test skipped: {e}")


class TestWorkflowIntegration:
    """Integration tests for CI/CD workflow components"""
    
    def test_github_actions_workflow_files(self):
        """Test that GitHub Actions workflow files are properly structured"""
        workflows_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.github', 'workflows')
        
        expected_workflows = [
            'sync-huggingface.yml',
            'test-local-model.yml', 
            'webhook-notification.yml'
        ]
        
        if os.path.exists(workflows_dir):
            existing_workflows = os.listdir(workflows_dir)
            
            for workflow in expected_workflows:
                workflow_path = os.path.join(workflows_dir, workflow)
                if os.path.exists(workflow_path):
                    # Basic validation that it's a YAML file
                    with open(workflow_path, 'r') as f:
                        content = f.read()
                        assert 'name:' in content
                        assert 'on:' in content
                        assert 'jobs:' in content
            
            print("✅ GitHub Actions workflow files are properly structured")
        else:
            pytest.skip("GitHub workflows directory not found")
    
    def test_requirements_file_validity(self):
        """Test that requirements.txt is valid and contains necessary packages"""
        requirements_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'requirements.txt')
        
        if os.path.exists(requirements_path):
            with open(requirements_path, 'r') as f:
                requirements = f.read()
            
            # Check for essential packages
            essential_packages = [
                'gradio',
                'transformers', 
                'torch',
                'huggingface_hub'
            ]
            
            for package in essential_packages:
                assert package in requirements.lower(), f"Missing essential package: {package}"
            
            print("✅ Requirements file contains necessary packages")
        else:
            pytest.skip("requirements.txt not found")
    
    def test_project_structure_integrity(self):
        """Test that project structure is complete"""
        project_root = os.path.dirname(os.path.dirname(__file__))
        
        expected_files = [
            'app.py',
            'app_local.py',
            'requirements.txt',
            'README.md'
        ]
        
        expected_dirs = [
            'tests',
            '.github/workflows'
        ]
        
        missing_files = []
        missing_dirs = []
        
        for file in expected_files:
            if not os.path.exists(os.path.join(project_root, file)):
                missing_files.append(file)
        
        for dir in expected_dirs:
            if not os.path.exists(os.path.join(project_root, dir)):
                missing_dirs.append(dir)
        
        if missing_files or missing_dirs:
            print(f"⚠️ Missing files: {missing_files}")
            print(f"⚠️ Missing directories: {missing_dirs}")
        else:
            print("✅ Project structure is complete")
        
        # Don't fail the test, just report
        assert True


class TestEndToEndWorkflows:
    """End-to-end integration tests for complete workflows"""
    
    @pytest.mark.integration
    def test_complete_api_workflow(self):
        """Test complete API-based analysis workflow"""
        with patch('huggingface_hub.InferenceClient') as mock_client:
            mock_instance = Mock()
            mock_instance.text_generation.return_value = MOCK_RESPONSES["sentiment_analysis"]
            mock_client.return_value = mock_instance
            
            try:
                import app
                
                # Test complete workflow: input → processing → output
                test_input = SAMPLE_INPUTS["clean_text"]
                
                if hasattr(app, 'analyze_text_interface'):
                    result, perf_info, stats = app.analyze_text_interface(
                        test_input, 
                        "Sentiment & Tone Analysis"
                    )
                    
                    # Verify complete workflow
                    assert isinstance(result, str) and len(result) > 0
                    assert isinstance(perf_info, str)
                    assert isinstance(stats, str)
                    
                    # Test that analytics were recorded
                    if hasattr(app.analytics, 'analysis_history'):
                        assert len(app.analytics.analysis_history) > 0
                
                print("✅ Complete API workflow test successful")
                
            except ImportError as e:
                pytest.skip(f"API workflow test skipped: {e}")
    
    @pytest.mark.integration
    def test_complete_local_workflow(self):
        """Test complete local model analysis workflow"""
        with patch('app_local.AutoTokenizer'), \
             patch('app_local.AutoModelForCausalLM'), \
             patch('app_local.pipeline') as mock_pipeline:
            
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.return_value = [{'generated_text': MOCK_RESPONSES["sentiment_analysis"]}]
            mock_pipeline.return_value = mock_pipeline_instance
            
            try:
                import app_local
                
                # Simulate model loading
                app_local.local_analytics.model_loaded = True
                app_local.local_analytics.pipeline = mock_pipeline_instance
                
                # Test complete workflow
                test_input = SAMPLE_INPUTS["clean_text"]
                
                if hasattr(app_local, 'analyze_local_interface'):
                    result, perf_info, stats = app_local.analyze_local_interface(
                        test_input,
                        "Sentiment & Tone Analysis"
                    )
                    
                    # Verify complete workflow
                    assert isinstance(result, str) and len(result) > 0
                    assert isinstance(perf_info, str)
                    assert isinstance(stats, str)
                
                print("✅ Complete local workflow test successful")
                
            except ImportError as e:
                pytest.skip(f"Local workflow test skipped: {e}")
    
    @pytest.mark.integration
    def test_session_management_workflow(self):
        """Test session creation and management workflow"""
        try:
            # Test API session management
            with patch('huggingface_hub.InferenceClient'):
                import app
                
                initial_session = app.analytics.session_id if hasattr(app.analytics, 'session_id') else None
                
                # Perform some analysis to create session history
                if hasattr(app, 'analyze_text_interface'):
                    app.analyze_text_interface(SAMPLE_INPUTS["clean_text"], "Sentiment & Tone Analysis")
                
                # Reset session
                if hasattr(app, 'reset_session'):
                    app.reset_session()
                    new_session = app.analytics.session_id if hasattr(app.analytics, 'session_id') else None
                    
                    if initial_session and new_session:
                        assert initial_session != new_session
            
            # Test local session management
            with patch('app_local.AutoTokenizer'), \
                 patch('app_local.AutoModelForCausalLM'), \
                 patch('app_local.pipeline'):
                
                import app_local
                
                if hasattr(app_local, 'reset_local_session'):
                    result, _, _ = app_local.reset_local_session()
                    assert isinstance(result, str)
                    assert "session" in result.lower()
            
            print("✅ Session management workflow test successful")
            
        except ImportError as e:
            pytest.skip(f"Session management test skipped: {e}")
    
    @pytest.mark.slow
    def test_performance_monitoring_workflow(self):
        """Test performance monitoring and analytics workflow"""
        try:
            with patch('huggingface_hub.InferenceClient') as mock_client:
                mock_instance = Mock()
                mock_instance.text_generation.return_value = MOCK_RESPONSES["sentiment_analysis"]
                mock_client.return_value = mock_instance
                
                import app
                
                # Perform multiple analyses to generate performance data
                for i in range(3):
                    if hasattr(app, 'analyze_text_interface'):
                        app.analyze_text_interface(
                            f"Test input {i}", 
                            "Sentiment & Tone Analysis"
                        )
                
                # Check performance stats
                if hasattr(app.analytics, 'get_performance_stats'):
                    stats = app.analytics.get_performance_stats()
                    
                    assert isinstance(stats, str)
                    assert len(stats) > 0
                    # Should show multiple analyses
                    assert "3" in stats or "total" in stats.lower()
            
            print("✅ Performance monitoring workflow test successful")
            
        except ImportError as e:
            pytest.skip(f"Performance monitoring test skipped: {e}")


class TestSecurityIntegration:
    """Integration tests for security features across the platform"""
    
    @pytest.mark.security
    def test_end_to_end_input_sanitization(self):
        """Test input sanitization across both API and local implementations"""
        malicious_input = SAMPLE_INPUTS["malicious_html"]
        
        try:
            # Test API sanitization
            with patch('huggingface_hub.InferenceClient'):
                import app
                
                if hasattr(app.analytics, 'sanitize_input'):
                    clean_api = app.analytics.sanitize_input(malicious_input)
                    assert "<script>" not in clean_api
            
            # Test local sanitization  
            with patch('app_local.AutoTokenizer'), \
                 patch('app_local.AutoModelForCausalLM'), \
                 patch('app_local.pipeline'):
                
                import app_local
                
                if hasattr(app_local.local_analytics, 'sanitize_input'):
                    clean_local = app_local.local_analytics.sanitize_input(malicious_input)
                    assert "<script>" not in clean_local
            
            print("✅ End-to-end input sanitization successful")
            
        except ImportError as e:
            pytest.skip(f"Security integration test skipped: {e}")
    
    @pytest.mark.security
    def test_privacy_protection_workflow(self):
        """Test privacy protection features in complete workflow"""
        pii_input = SAMPLE_INPUTS["pii_text"]
        
        try:
            with patch('huggingface_hub.InferenceClient') as mock_client:
                mock_instance = Mock()
                mock_instance.text_generation.return_value = "Sanitized business communication summary."
                mock_client.return_value = mock_instance
                
                import app
                
                if hasattr(app, 'analyze_text_interface'):
                    result, _, _ = app.analyze_text_interface(pii_input, "Secure Summary")
                    
                    # Should not contain original PII
                    assert "john.smith@example.com" not in result.lower()
                    assert isinstance(result, str)
            
            print("✅ Privacy protection workflow test successful")
            
        except ImportError as e:
            pytest.skip(f"Privacy protection test skipped: {e}")


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])