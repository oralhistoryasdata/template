import sys
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.llm_client import LLMClient
from models.structured_outputs import InterviewSummary, QualityScore

class InterviewSummarizer:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def summarize_interview(self, transcript: str, interview_metadata: dict) -> InterviewSummary:
        messages = [{
            "role": "user",
            "content": f"""
            Please create a summary of this oral history interview transcript:
            
            Interview Details:
            - Date: {interview_metadata.get('date', 'Unknown')}
            - Interviewer: {interview_metadata.get('interviewer', 'Unknown')}
            - Subject: {interview_metadata.get('subject', 'Unknown')}
            
            Transcript:
            {transcript[:8000]}  # Truncate for token limits
            
            Focus on key themes, significant events, and notable perspectives shared.
            """
        }]
        
        return self.llm.generate_structured(messages, InterviewSummary)
    
    def evaluate_summary_quality(self, original_transcript: str, summary: InterviewSummary) -> QualityScore:
        messages = [{
            "role": "user", 
            "content": f"""
            Evaluate if this summary adequately captures the key content of the original transcript:
            
            Original (first 2000 chars): {original_transcript[:2000]}
            
            Summary: {summary.summary}
            Key Themes: {', '.join(summary.key_themes)}
            
            Rate 1 if the summary is comprehensive and accurate, 0 if it needs improvement.
            """
        }]
        
        return self.llm.generate_structured(messages, QualityScore)