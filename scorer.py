"""
Multi-Dimensional Quality Scorer for Structured Outputs

Scoring dimensions:
- Completeness (0.30): How complete is the submission
- Format Compliance (0.20): Does it follow the expected format
- Coverage (0.25): How well does it cover the required aspects
- Clarity (0.15): How clear and readable is the content
- Validity (0.10): Is the content valid/accurate

Output: {weighted_score, quality_rating, scores: {dim: float}, feedback: [str], pass_threshold: bool}
"""

import json
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class FormatType(Enum):
    JSON = "json"
    MARKDOWN = "markdown"
    CODE = "code"
    TEXT = "text"
    XML = "xml"
    UNKNOWN = "unknown"


class QualityRating(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAIL = "fail"


@dataclass
class ScoringResult:
    weighted_score: float
    quality_rating: QualityRating
    scores: Dict[str, float]
    feedback: List[str]
    pass_threshold: bool
    format_detected: FormatType


class QualityScorer:
    """Multi-dimensional quality scorer for structured outputs."""
    
    # Dimension weights
    WEIGHTS = {
        "completeness": 0.30,
        "format_compliance": 0.20,
        "coverage": 0.25,
        "clarity": 0.15,
        "validity": 0.10
    }
    
    # Thresholds
    PASS_THRESHOLD = 0.6
    EXCELLENT_THRESHOLD = 0.85
    GOOD_THRESHOLD = 0.75
    ACCEPTABLE_THRESHOLD = 0.6
    
    def __init__(self, expected_schema: Optional[Dict] = None):
        self.expected_schema = expected_schema or {}
    
    def detect_format(self, content: str) -> FormatType:
        """Auto-detect the format of the content."""
        content = content.strip()
        
        # Try JSON
        if content.startswith('{') or content.startswith('['):
            try:
                json.loads(content)
                return FormatType.JSON
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Try XML
        if content.startswith('<?xml') or content.startswith('<'):
            try:
                ET.fromstring(content)
                return FormatType.XML
            except ET.ParseError:
                pass
        
        # Check for code patterns
        code_patterns = [
            r'^def\s+\w+\s*\(',
            r'^class\s+\w+',
            r'^import\s+\w+',
            r'^from\s+\w+\s+import',
            r'^function\s+\w+\s*\(',
            r'^const\s+\w+\s*=',
            r'^let\s+\w+\s*=',
            r'^var\s+\w+\s*=',
            r'^\s*#include',
            r'^\s*package\s+\w+',
            r'^\s*public\s+(class|void|static)',
            r'^\s*private\s+(class|void|static)',
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, content, re.MULTILINE):
                return FormatType.CODE
        
        # Check for markdown
        md_patterns = [
            r'^#{1,6}\s+.+$',
            r'^\*\*[^*]+\*\*',
            r'^\*[^*]+\*',
            r'^```\w*',
            r'^\[.+\]\(.+\)',
            r'^\|.*\|',
        ]
        
        md_count = 0
        for pattern in md_patterns:
            if re.search(pattern, content, re.MULTILINE):
                md_count += 1
        
        if md_count >= 2:
            return FormatType.MARKDOWN
        
        # Default to text
        return FormatType.TEXT
    
    def score_completeness(self, content: str, format_type: FormatType) -> Tuple[float, List[str]]:
        """Score completeness (0-1)."""
        feedback = []
        score = 0.5  # Base score
        
        if not content or len(content.strip()) == 0:
            return 0.0, ["Content is empty"]
        
        # Length-based scoring
        length = len(content.strip())
        
        if format_type == FormatType.JSON:
            try:
                data = json.loads(content)
                # Check if it's a dict with multiple keys
                if isinstance(data, dict):
                    key_count = len(data.keys())
                    if key_count >= 5:
                        score = 1.0
                        feedback.append("JSON has 5+ fields")
                    elif key_count >= 3:
                        score = 0.8
                        feedback.append("JSON has 3-4 fields")
                    elif key_count >= 1:
                        score = 0.6
                        feedback.append("JSON has 1-2 fields")
                elif isinstance(data, list):
                    if len(data) >= 10:
                        score = 1.0
                        feedback.append("JSON array has 10+ items")
                    elif len(data) >= 5:
                        score = 0.8
                        feedback.append("JSON array has 5-9 items")
                    elif len(data) >= 1:
                        score = 0.6
                        feedback.append("JSON array has 1-4 items")
            except json.JSONDecodeError:
                score = 0.1
                feedback.append("Invalid JSON")
        
        elif format_type == FormatType.MARKDOWN:
            # Check for common markdown elements
            has_headers = bool(re.search(r'^#{1,6}\s+', content, re.MULTILINE))
            has_bold = bool(re.search(r'\*\*[^*]+\*\*', content))
            has_list = bool(re.search(r'^[\s]*[-*+]\s+', content, re.MULTILINE))
            has_code = bool(re.search(r'```[\s\S]*?```', content))
            
            element_count = sum([has_headers, has_bold, has_list, has_code])
            score = min(1.0, 0.4 + element_count * 0.15)
            
            if has_headers:
                feedback.append("Has headers")
            if has_list:
                feedback.append("Has lists")
            if has_bold:
                feedback.append("Has bold text")
            if has_code:
                feedback.append("Has code blocks")
        
        elif format_type == FormatType.CODE:
            # Check for function/class definitions, imports, etc
            has_imports = bool(re.search(r'^(import|from)\s+', content, re.MULTILINE))
            has_functions = bool(re.search(r'(def|function|fn)\s+\w+', content))
            has_classes = bool(re.search(r'class\s+\w+', content))
            has_docstring = bool(re.search(r'(""".*?"""|\'\'\'.*?\'\'\')', content, re.DOTALL))
            
            element_count = sum([has_imports, has_functions, has_classes, has_docstring])
            score = min(1.0, 0.3 + element_count * 0.175)
            
            if has_functions:
                feedback.append("Has functions/methods")
            if has_classes:
                feedback.append("Has class definitions")
            if has_docstring:
                feedback.append("Has documentation")
        
        else:  # TEXT
            # Word count based scoring
            word_count = len(content.split())
            if word_count >= 500:
                score = 1.0
                feedback.append("Comprehensive text (500+ words)")
            elif word_count >= 200:
                score = 0.8
                feedback.append("Substantial text (200-499 words)")
            elif word_count >= 100:
                score = 0.6
                feedback.append("Moderate text (100-199 words)")
            elif word_count >= 50:
                score = 0.5
                feedback.append("Brief text (50-99 words)")
            else:
                score = 0.4
                feedback.append("Short text (<50 words)")
        
        return score, feedback
    
    def score_format_compliance(self, content: str, format_type: FormatType) -> Tuple[float, List[str]]:
        """Score format compliance (0-1)."""
        feedback = []
        score = 0.5
        
        if format_type == FormatType.JSON:
            try:
                json.loads(content)
                score = 1.0
                feedback.append("Valid JSON format")
            except json.JSONDecodeError as e:
                score = 0.0
                feedback.append(f"Invalid JSON: {str(e)}")
        
        elif format_type == FormatType.XML:
            try:
                ET.fromstring(content)
                score = 1.0
                feedback.append("Valid XML format")
            except ET.ParseError as e:
                score = 0.0
                feedback.append(f"Invalid XML: {str(e)}")
        
        elif format_type == FormatType.MARKDOWN:
            # Check for consistent formatting
            lines = content.split('\n')
            empty_lines = sum(1 for l in lines if not l.strip())
            total_lines = len(lines)
            
            # Consistency score
            consistency = 1.0 - (empty_lines / max(total_lines, 1))
            
            # Check balanced formatting
            bold_open = len(re.findall(r'\*\*[^*]+\*\*', content))
            bold_unbalanced = abs(content.count('**') - bold_open * 2)
            
            if bold_unbalanced == 0:
                score = 1.0
                feedback.append("Well-formatted markdown")
            elif bold_unbalanced <= 2:
                score = 0.7
                feedback.append("Mostly consistent formatting")
            else:
                score = 0.5
                feedback.append("Inconsistent formatting")
        
        elif format_type == FormatType.CODE:
            # Check for consistent indentation
            lines = content.split('\n')
            indent_patterns = []
            for line in lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    indent_patterns.append(indent)
            
            if indent_patterns:
                unique_indents = len(set(indent_patterns))
                if unique_indents <= 3:
                    score = 1.0
                    feedback.append("Consistent indentation")
                elif unique_indents <= 5:
                    score = 0.7
                    feedback.append("Moderate indentation variety")
                else:
                    score = 0.5
                    feedback.append("Inconsistent indentation")
            
            # Check for semicolons (for JS) or brackets balance
            if content.count('{') == content.count('}'):
                if score == 1.0:
                    feedback.append("Balanced brackets")
            else:
                score = min(score, 0.6)
                feedback.append("Unbalanced brackets")
        
        else:  # TEXT
            # Check for proper punctuation and capitalization
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if sentences:
                # Check capitalization
                capitalized = sum(1 for s in sentences if s[0].isupper())
                cap_ratio = capitalized / len(sentences)
                
                # Check ending punctuation
                ended = sum(1 for s in sentences if s[-1] in '.!?')
                end_ratio = ended / len(sentences)
                
                score = (cap_ratio + end_ratio) / 2
                
                if score >= 0.9:
                    feedback.append("Well-formatted text")
                elif score >= 0.7:
                    feedback.append("Mostly formatted text")
        
        return score, feedback
    
    def score_coverage(self, content: str, format_type: FormatType, schema: Optional[Dict] = None) -> Tuple[float, List[str]]:
        """Score coverage of required aspects (0-1)."""
        feedback = []
        score = 0.5
        
        schema = schema or self.expected_schema
        
        if format_type == FormatType.JSON and schema:
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    required_fields = schema.get('required', [])
                    present_fields = set(data.keys())
                    
                    covered = sum(1 for f in required_fields if f in present_fields)
                    coverage = covered / len(required_fields) if required_fields else 1.0
                    
                    score = coverage
                    
                    if coverage == 1.0:
                        feedback.append("All required fields present")
                    elif coverage >= 0.5:
                        feedback.append(f"Partial coverage: {covered}/{len(required_fields)} fields")
                    else:
                        feedback.append(f"Low coverage: {covered}/{len(required_fields)} fields")
                else:
                    score = 0.3
                    feedback.append("Expected object, got array")
            except:
                score = 0.0
                feedback.append("Unable to parse JSON for coverage check")
        
        elif format_type == FormatType.MARKDOWN:
            # Check coverage of common sections
            sections = {
                'title': r'^#{1,3}\s+.+',
                'introduction': r'(intro|introduction|overview)',
                'body': r'#{1,3}\s+',
                'conclusion': r'(conclusion|summary|final)',
                'list': r'^[\s]*[-*+]\s+',
            }
            
            covered = sum(1 for _, pattern in sections.items() 
                        if re.search(pattern, content, re.IGNORECASE | re.MULTILINE))
            
            score = covered / len(sections)
            
            if covered >= 4:
                feedback.append("Comprehensive markdown structure")
            elif covered >= 2:
                feedback.append("Basic structure present")
            else:
                feedback.append("Limited structure")
        
        elif format_type == FormatType.CODE:
            # Check for common code elements
            elements = {
                'imports': r'^(import|from)\s+',
                'functions': r'(def|function|fn)\s+\w+',
                'classes': r'class\s+\w+',
                'docs': r'(""".*?"""|\'\'\'.*?\'\'\')',
                'type_hints': r':\s*(int|str|bool|float|list|dict|Optional)',
            }
            
            covered = sum(1 for _, pattern in elements.items()
                        if re.search(pattern, content, re.MULTILINE))
            
            score = covered / len(elements)
            
            if covered >= 4:
                feedback.append("Well-documented code")
            elif covered >= 2:
                feedback.append("Basic code structure")
            else:
                feedback.append("Minimal code structure")
        
        else:  # TEXT
            # Check for multiple paragraphs
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            if len(paragraphs) >= 5:
                score = 1.0
                feedback.append("Multiple well-developed paragraphs")
            elif len(paragraphs) >= 3:
                score = 0.8
                feedback.append("Good paragraph structure")
            elif len(paragraphs) >= 2:
                score = 0.6
                feedback.append("Basic paragraph structure")
            else:
                score = 0.4
                feedback.append("Single paragraph")
        
        return score, feedback
    
    def score_clarity(self, content: str, format_type: FormatType) -> Tuple[float, List[str]]:
        """Score clarity and readability (0-1)."""
        feedback = []
        score = 0.5
        
        # Remove extra whitespace
        cleaned = ' '.join(content.split())
        words = cleaned.split()
        
        if not words:
            return 0.0, ["No content to evaluate"]
        
        # Sentence detection
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)
        
        # Average sentence length
        if sentence_count > 0:
            avg_sentence_len = len(words) / sentence_count
        else:
            avg_sentence_len = len(words)
        
        # Optimal sentence length is 10-20 words
        if 10 <= avg_sentence_len <= 20:
            sentence_score = 1.0
            feedback.append("Optimal sentence length")
        elif 5 <= avg_sentence_len < 10:
            sentence_score = 0.8
            feedback.append("Short sentences")
        elif 20 < avg_sentence_len <= 30:
            sentence_score = 0.7
            feedback.append("Longer sentences")
        else:
            sentence_score = 0.5
            feedback.append("Very short or very long sentences")
        
        # Check for clarity issues
        issues = []
        
        # Very long words (potential issues)
        long_words = [w for w in words if len(w) > 20]
        if len(long_words) > len(words) * 0.1:
            issues.append("Many long words")
        
        # Repetitive words
        word_freq = {}
        for w in words:
            w_lower = w.lower()
            word_freq[w_lower] = word_freq.get(w_lower, 0) + 1
        
        repetitive = [(w, c) for w, c in word_freq.items() if c > 5]
        if repetitive:
            issues.append("Repetitive words detected")
        
        # Calculate clarity score
        issue_penalty = len(issues) * 0.1
        clarity_score = sentence_score - issue_penalty
        
        score = max(0.0, min(1.0, clarity_score))
        
        if issues:
            feedback.extend(issues)
        
        if score >= 0.8:
            feedback.insert(0, "Very clear content")
        elif score >= 0.6:
            feedback.insert(0, "Generally clear content")
        
        return score, feedback
    
    def score_validity(self, content: str, format_type: FormatType) -> Tuple[float, List[str]]:
        """Score validity/accuracy (0-1)."""
        feedback = []
        score = 0.7  # Default assumption of validity
        
        if format_type == FormatType.JSON:
            try:
                data = json.loads(content)
                
                # Check for null values in expected places
                null_count = str(data).count('null')
                if null_count > 5:
                    score -= 0.2
                    feedback.append(f"Contains {null_count} null values")
                
                # Check for empty strings
                empty_count = str(data).count('""')
                if empty_count > 3:
                    score -= 0.1
                    feedback.append("Contains empty string values")
                
                # Check for reasonable values
                if isinstance(data, dict):
                    non_empty = sum(1 for v in data.values() if v not in [None, "", [], {}])
                    if non_empty > 0:
                        score = 0.9
                        feedback.append("Valid JSON structure")
                
            except json.JSONDecodeError:
                score = 0.0
                feedback.append("Invalid JSON")
        
        elif format_type == FormatType.CODE:
            # Basic syntax checks
            # Check for common syntax errors
            
            # Unclosed brackets
            open_braces = content.count('{')
            close_braces = content.count('}')
            if open_braces != close_braces:
                score -= 0.3
                feedback.append("Unbalanced braces")
            
            # Unclosed parens
            open_parens = content.count('(')
            close_parens = content.count(')')
            if open_parens != close_parens:
                score -= 0.2
                feedback.append("Unbalanced parentheses")
            
            # Check for TODO/FIXME (indicates incomplete code)
            if re.search(r'\b(TODO|FIXME|HACK|XXX)\b', content):
                score -= 0.1
                feedback.append("Contains incomplete markers")
            
            if score >= 0.7:
                feedback.append("Valid code structure")
        
        elif format_type == FormatType.MARKDOWN:
            # Check for broken links
            broken_links = re.findall(r'\[([^\]]+)\]\(\)', content)
            if broken_links:
                score -= 0.2
                feedback.append(f"{len(broken_links)} empty/broken links")
            
            # Check for image placeholders
            img_empty = re.findall(r'!\[([^\]]*)\]\(\)', content)
            if img_empty:
                score -= 0.1
                feedback.append("Contains image placeholders")
            
            if score >= 0.7:
                feedback.append("Valid markdown")
        
        else:  # TEXT
            # Check for placeholder text
            placeholders = [
                r'\[TODO\]',
                r'\[INSERT.*\]',
                r'\{.*\}',
                r'Lorem ipsum',
            ]
            
            placeholder_count = 0
            for pattern in placeholders:
                placeholder_count += len(re.findall(pattern, content, re.IGNORECASE))
            
            if placeholder_count > 0:
                score -= 0.3
                feedback.append(f"Contains {placeholder_count} placeholders")
            
            if score >= 0.7:
                feedback.append("Valid text content")
        
        score = max(0.0, min(1.0, score))
        return score, feedback
    
    def score(self, content: str, schema: Optional[Dict] = None) -> ScoringResult:
        """
        Score content across all dimensions.
        
        Args:
            content: The content to score
            schema: Optional schema for validation
            
        Returns:
            ScoringResult with weighted_score, quality_rating, scores, feedback, pass_threshold
        """
        # Detect format
        format_type = self.detect_format(content)
        
        # Score each dimension
        completeness_score, completeness_feedback = self.score_completeness(content, format_type)
        format_score, format_feedback = self.score_format_compliance(content, format_type)
        coverage_score, coverage_feedback = self.score_coverage(content, format_type, schema)
        clarity_score, clarity_feedback = self.score_clarity(content, format_type)
        validity_score, validity_feedback = self.score_validity(content, format_type)
        
        # Store individual scores
        scores = {
            "completeness": completeness_score,
            "format_compliance": format_score,
            "coverage": coverage_score,
            "clarity": clarity_score,
            "validity": validity_score
        }
        
        # Calculate weighted score
        weighted_score = sum(
            score * self.WEIGHTS[dim]
            for dim, score in scores.items()
        )
        
        # Determine quality rating
        if weighted_score >= self.EXCELLENT_THRESHOLD:
            quality_rating = QualityRating.EXCELLENT
        elif weighted_score >= self.GOOD_THRESHOLD:
            quality_rating = QualityRating.GOOD
        elif weighted_score >= self.ACCEPTABLE_THRESHOLD:
            quality_rating = QualityRating.ACCEPTABLE
        elif weighted_score >= 0.4:
            quality_rating = QualityRating.POOR
        else:
            quality_rating = QualityRating.FAIL
        
        # Collect all feedback
        all_feedback = (
            completeness_feedback +
            format_feedback +
            coverage_feedback +
            clarity_feedback +
            validity_feedback
        )
        
        # Deduplicate feedback
        all_feedback = list(dict.fromkeys(all_feedback))
        
        # Determine pass/fail
        pass_threshold = weighted_score >= self.PASS_THRESHOLD
        
        return ScoringResult(
            weighted_score=round(weighted_score, 3),
            quality_rating=quality_rating,
            scores={k: round(v, 3) for k, v in scores.items()},
            feedback=all_feedback[:10],  # Limit to top 10 feedback items
            pass_threshold=pass_threshold,
            format_detected=format_type
        )
    
    def score_batch(self, contents: List[str], schema: Optional[Dict] = None) -> List[ScoringResult]:
        """
        Score multiple submissions.
        
        Args:
            contents: List of content strings to score
            schema: Optional schema for validation
            
        Returns:
            List of ScoringResult objects
        """
        return [self.score(content, schema) for content in contents]


# CLI interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python scorer.py <content_file> [schema_file]")
        sys.exit(1)
    
    # Load content
    with open(sys.argv[1], 'r') as f:
        content = f.read()
    
    # Load schema if provided
    schema = None
    if len(sys.argv) >= 3:
        with open(sys.argv[2], 'r') as f:
            schema = json.load(f)
    
    # Score
    scorer = QualityScorer()
    result = scorer.score(content, schema)
    
    # Output
    output = {
        "weighted_score": result.weighted_score,
        "quality_rating": result.quality_rating.value,
        "scores": result.scores,
        "feedback": result.feedback,
        "pass_threshold": result.pass_threshold,
        "format_detected": result.format_detected.value
    }
    
    print(json.dumps(output, indent=2))
