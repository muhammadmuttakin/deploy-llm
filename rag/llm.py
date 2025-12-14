import time
import google.generativeai as genai
from config import MODEL_NAME, GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)


def ask_llm(prompt: str, system_prompt: str = None, max_retries: int = 3):
    """
    Call Gemini API dengan retry logic untuk handle transient errors.
    
    Args:
        prompt: Main user prompt
        system_prompt: System instruction (optional)
        max_retries: Max retry attempts for transient errors
    
    Returns:
        Response text from LLM
    
    Raises:
        ValueError: Jika response di-block oleh safety filter
        RuntimeError: Jika semua retry attempts gagal
    """
    
    # Build full prompt with system instruction
    full_prompt = ""
    if system_prompt:
        full_prompt += f"### SYSTEM INSTRUCTION ###\n{system_prompt}\n\n"
    full_prompt += prompt

    messages = [
        {
            "role": "user",
            "parts": [{"text": full_prompt}]
        }
    ]

    for attempt in range(max_retries):
        try:
            response = model.generate_content(messages)
            return response.text
        
        except genai.types.StopCandidateException as e:
            """Response blocked by safety filter"""
            print(f"[LLM Warning] Response blocked by safety filter: {e}")
            raise ValueError(
                "Response tidak bisa diproses karena melanggar content policy. "
                "Mohon ajukan pertanyaan yang berbeda."
            )
        
        except genai.types.APIError as e:
            error_str = str(e).lower()
            
            # Handle rate limiting
            if "429" in str(e) or "rate_limit" in error_str or "quota" in error_str:
                wait_time = (2 ** attempt) + 1  # Exponential backoff: 2, 5, 10 seconds
                print(f"[LLM] Rate limited (attempt {attempt + 1}/{max_retries}). "
                      f"Menunggu {wait_time} detik sebelum retry...")
                time.sleep(wait_time)
                continue
            
            # Handle timeout/connection errors (retryable)
            elif "timeout" in error_str or "connection" in error_str or "503" in str(e):
                wait_time = (2 ** attempt) + 1
                print(f"[LLM] Connection error (attempt {attempt + 1}/{max_retries}). "
                      f"Retrying dalam {wait_time} detik...")
                time.sleep(wait_time)
                continue
            
            # Non-retryable API error
            else:
                print(f"[LLM Error] API Error: {e}")
                raise
        
        except Exception as e:
            """Unexpected errors"""
            print(f"[LLM Error] Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
            
            # Last attempt - jangan retry lagi
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Failed to get response from LLM after {max_retries} attempts. "
                    f"Error: {str(e)}"
                )
            
            # Retry dengan delay
            time.sleep(1)
    
    # Should not reach here, but just in case
    raise RuntimeError(f"Max retries ({max_retries}) exceeded without success")