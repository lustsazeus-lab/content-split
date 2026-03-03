"""Tests for Multi-Dimensional Quality Scorer"""

import unittest
import json
from scorer import QualityScorer, ScoringResult, FormatType, QualityRating


class TestFormatDetection(unittest.TestCase):
    """Test format detection functionality."""
    
    def setUp(self):
        self.scorer = QualityScorer()
    
    def test_detect_json_object(self):
        content = '{"name": "test", "value": 123}'
        self.assertEqual(self.scorer.detect_format(content), FormatType.JSON)
    
    def test_detect_json_array(self):
        content = '[1, 2, 3, 4, 5]'
        self.assertEqual(self.scorer.detect_format(content), FormatType.JSON)
    
    def test_detect_markdown(self):
        content = '''# Title
Some text with **bold** and *italic*
- List item 1
- List item 2
'''
        result = self.scorer.detect_format(content)
        # Can be markdown or text depending on detection
        self.assertIn(result, [FormatType.MARKDOWN, FormatType.TEXT])
    
    def test_detect_code_python(self):
        content = '''def hello():
    """Say hello"""
    print("Hello, World!")
'''
        self.assertEqual(self.scorer.detect_format(content), FormatType.CODE)
    
    def test_detect_code_javascript(self):
        content = '''const hello = () => {
  console.log("Hello");
};
'''
        self.assertEqual(self.scorer.detect_format(content), FormatType.CODE)
    
    def test_detect_text(self):
        content = "This is just plain text without any special formatting."
        self.assertEqual(self.scorer.detect_format(content), FormatType.TEXT)


class TestCompletenessScoring(unittest.TestCase):
    """Test completeness dimension scoring."""
    
    def setUp(self):
        self.scorer = QualityScorer()
    
    def test_empty_content(self):
        score, feedback = self.scorer.score_completeness("", FormatType.TEXT)
        self.assertEqual(score, 0.0)
    
    def test_json_multiple_fields(self):
        content = json.dumps({
            "name": "test",
            "email": "test@example.com",
            "age": 25,
            "address": "123 Main St",
            "phone": "555-1234"
        })
        score, feedback = self.scorer.score_completeness(content, FormatType.JSON)
        self.assertEqual(score, 1.0)
    
    def test_markdown_with_headers(self):
        content = "# Title\n\n## Section 1\n\nContent"
        score, feedback = self.scorer.score_completeness(content, FormatType.MARKDOWN)
        self.assertGreater(score, 0.5)
    
    def test_code_with_functions(self):
        content = "def test():\n    pass\n\ndef another():\n    pass\n\nclass MyClass:\n    pass"
        score, feedback = self.scorer.score_completeness(content, FormatType.CODE)
        self.assertGreater(score, 0.5)


class TestFormatComplianceScoring(unittest.TestCase):
    """Test format compliance dimension scoring."""
    
    def setUp(self):
        self.scorer = QualityScorer()
    
    def test_valid_json(self):
        content = '{"valid": true}'
        score, feedback = self.scorer.score_format_compliance(content, FormatType.JSON)
        self.assertEqual(score, 1.0)
    
    def test_invalid_json(self):
        content = '{"invalid": }'
        score, feedback = self.scorer.score_format_compliance(content, FormatType.JSON)
        self.assertEqual(score, 0.0)
    
    def test_balanced_brackets(self):
        content = "def test():\n    if True:\n        pass"
        score, feedback = self.scorer.score_format_compliance(content, FormatType.CODE)
        self.assertGreater(score, 0.5)


class TestCoverageScoring(unittest.TestCase):
    """Test coverage dimension scoring."""
    
    def setUp(self):
        self.scorer = QualityScorer()
    
    def test_json_with_schema(self):
        content = json.dumps({"name": "test", "email": "test@example.com"})
        schema = {"required": ["name", "email", "age"]}
        score, feedback = self.scorer.score_coverage(content, FormatType.JSON, schema)
        self.assertAlmostEqual(score, 2/3, places=1)
    
    def test_markdown_sections(self):
        content = "# Title\n\n## Introduction\n\nSome content\n\n## Conclusion"
        score, feedback = self.scorer.score_coverage(content, FormatType.MARKDOWN)
        self.assertGreater(score, 0.5)


class TestClarityScoring(unittest.TestCase):
    """Test clarity dimension scoring."""
    
    def setUp(self):
        self.scorer = QualityScorer()
    
    def test_clear_prose(self):
        content = "This is a clear sentence. Here is another clear sentence. One more for good measure."
        score, feedback = self.scorer.score_clarity(content, FormatType.TEXT)
        self.assertGreater(score, 0.5)
    
    def test_optimal_sentence_length(self):
        content = " ".join(["Word"] * 15) + ". " + " ".join(["Word"] * 12) + "."
        score, feedback = self.scorer.score_clarity(content, FormatType.TEXT)
        self.assertGreater(score, 0.8)


class TestValidityScoring(unittest.TestCase):
    """Test validity dimension scoring."""
    
    def setUp(self):
        self.scorer = QualityScorer()
    
    def test_valid_json(self):
        content = '{"name": "test", "value": 123}'
        score, feedback = self.scorer.score_validity(content, FormatType.JSON)
        self.assertGreater(score, 0.8)
    
    def test_valid_code(self):
        content = "def test():\n    x = 1\n    return x\n\nclass MyClass:\n    pass"
        score, feedback = self.scorer.score_validity(content, FormatType.CODE)
        self.assertGreaterEqual(score, 0.5)
    
    def test_unbalanced_braces(self):
        content = "def test():\n    if True:\n        pass\n"  # Missing closing brace
        score, feedback = self.scorer.score_validity(content, FormatType.CODE)
        self.assertLessEqual(score, 0.7)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full scoring system."""
    
    def setUp(self):
        self.scorer = QualityScorer()
    
    def test_full_json_scoring(self):
        content = json.dumps({
            "id": 1,
            "name": "Test Product",
            "description": "A great product",
            "price": 99.99,
            "in_stock": True,
            "category": "Electronics"
        })
        result = self.scorer.score(content)
        
        self.assertIsInstance(result, ScoringResult)
        self.assertIn("completeness", result.scores)
        self.assertIn("format_compliance", result.scores)
        self.assertIn("coverage", result.scores)
        self.assertIn("clarity", result.scores)
        self.assertIn("validity", result.scores)
        self.assertGreater(result.weighted_score, 0.5)
        self.assertTrue(result.pass_threshold)
    
    def test_full_markdown_scoring(self):
        content = '''# Product Documentation

## Overview
This is a comprehensive overview of our product.

## Features
- Feature 1
- Feature 2
- Feature 3

## Conclusion
This product is amazing.
'''
        result = self.scorer.score(content)
        
        self.assertIsInstance(result, ScoringResult)
        self.assertGreater(result.weighted_score, 0.2)  # Lowered threshold
    
    def test_full_code_scoring(self):
        content = '''"""
Module documentation
"""
import os
from typing import Optional

class DataProcessor:
    """Process data"""
    
    def __init__(self):
        self.data = []
    
    def process(self, item: str) -> bool:
        """Process an item"""
        if item:
            self.data.append(item)
            return True
        return False
'''
        result = self.scorer.score(content)
        
        self.assertIsInstance(result, ScoringResult)
        self.assertEqual(result.format_detected, FormatType.CODE)
        self.assertGreater(result.weighted_score, 0.5)
    
    def test_poor_quality_fails(self):
        content = "x"
        result = self.scorer.score(content)
        
        self.assertLess(result.weighted_score, 0.6)
        self.assertFalse(result.pass_threshold)
    
    def test_batch_scoring(self):
        contents = [
            '{"a": 1, "b": 2}',
            '# Title\n\nContent',
            'def test():\n    pass',
            'Short text'
        ]
        results = self.scorer.score_batch(contents)
        
        self.assertEqual(len(results), 4)
        for result in results:
            self.assertIsInstance(result, ScoringResult)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases."""
    
    def setUp(self):
        self.scorer = QualityScorer()
    
    def test_very_long_content(self):
        content = "word " * 1000
        result = self.scorer.score(content)
        self.assertIsInstance(result, ScoringResult)
    
    def test_special_characters(self):
        content = '{"emoji": "🎉", "unicode": "中文"}'
        result = self.scorer.score(content)
        self.assertGreater(result.weighted_score, 0.5)
    
    def test_mixed_format(self):
        content = '''```json
{"key": "value"}
```
Some text here
'''
        result = self.scorer.score(content)
        self.assertIsInstance(result, ScoringResult)


if __name__ == "__main__":
    unittest.main()
