import time
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
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
        
        except ValueError as e:
            """Response blocked by safety filter or empty response"""
            error_str = str(e).lower()
            if "block" in error_str or "safety" in error_str or "finish_reason" in error_str:
                print(f"[LLM Warning] Response blocked by safety filter: {e}")
                raise ValueError(
                    "Response tidak bisa diproses karena melanggar content policy. "
                    "Mohon ajukan pertanyaan yang berbeda."
                )
            else:
                # Re-raise jika bukan safety issue
                raise
        
        except google_exceptions.ResourceExhausted as e:
            """Rate limiting / Quota exceeded"""
            wait_time = (2 ** attempt) + 1  # Exponential backoff: 2, 5, 10 seconds
            print(f"[LLM] Rate limited (attempt {attempt + 1}/{max_retries}). "
                  f"Menunggu {wait_time} detik sebelum retry...")
            
            if attempt == max_retries - 1:
                raise RuntimeError(
                    "API quota exceeded atau rate limit. Silakan coba beberapa saat lagi."
                )
            time.sleep(wait_time)
            continue
        
        except (google_exceptions.ServiceUnavailable, 
                google_exceptions.DeadlineExceeded,
                google_exceptions.InternalServerError) as e:
            """Timeout/Connection/503 errors (retryable)"""
            wait_time = (2 ** attempt) + 1
            print(f"[LLM] Connection error (attempt {attempt + 1}/{max_retries}). "
                  f"Retrying dalam {wait_time} detik...")
            
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Tidak dapat terhubung ke server AI setelah {max_retries} percobaan. "
                    f"Silakan coba lagi nanti."
                )
            time.sleep(wait_time)
            continue
        
        except google_exceptions.GoogleAPIError as e:
            """General Google API errors"""
            error_str = str(e).lower()
            
            # Check if it's a retryable error
            if "timeout" in error_str or "connection" in error_str or "503" in error_str:
                wait_time = (2 ** attempt) + 1
                print(f"[LLM] API error (attempt {attempt + 1}/{max_retries}). "
                      f"Retrying dalam {wait_time} detik...")
                
                if attempt == max_retries - 1:
                    raise RuntimeError(f"API error setelah {max_retries} percobaan: {str(e)}")
                time.sleep(wait_time)
                continue
            
            # Non-retryable API error
            print(f"[LLM Error] Non-retryable API Error: {e}")
            raise RuntimeError(f"API Error: {str(e)}")
        
        except Exception as e:
            """Unexpected errors"""
            error_str = str(e).lower()
            print(f"[LLM Error] Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
            
            # Check if error message indicates a retryable issue
            if any(keyword in error_str for keyword in ["timeout", "connection", "503", "502", "500"]):
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt + 1)
                    continue
            
            # Last attempt or non-retryable error
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Failed to get response from LLM after {max_retries} attempts. "
                    f"Error: {str(e)}"
                )
            
            # Retry dengan delay
            time.sleep(1)
    
    # Should not reach here, but just in case
    raise RuntimeError(f"Max retries ({max_retries}) exceeded without success")